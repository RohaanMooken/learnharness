/**
 * GitHub sync, the transport (fork addition, 2026-11-02): a thin client over
 * GitHub's REST Git Data API, so every platform (Windows, macOS, Linux, the
 * Android guest) shares one code path and needs no git binary. It knows
 * nothing about LearnHarness files: it lists a branch, reads blobs and
 * pushes one atomic commit.
 *
 * Why the Git Data API and not the contents endpoints: one push is ONE
 * commit however many files it touches, and creating a text file through
 * a tree entry's inline `content` costs no request of its own. Blob POSTs
 * count against GitHub's content-creation limits (about 80 a minute), and a
 * first sync can carry hundreds of chats. Only binary files (photo
 * sketches) take the blob endpoint.
 *
 * The auth is a fine-grained PAT (Contents read/write on the one repo). The
 * token is only ever sent to api.github.com in a Bearer header.
 */

export type GitHubErrorCode =
	/** 401, or 403 that is not a rate limit: the token is bad, expired, or lacks Contents write. */
	| "auth"
	/** 404: the repo or branch is wrong, or the token cannot see a private repo. */
	| "not-found"
	/** 403/429 with the limit exhausted. `resetAt` (epoch ms) says when it lifts. */
	| "rate-limit"
	/** The branch moved while pushing (not a fast-forward): pull again and retry. */
	| "conflict"
	/** 409: the repo has no commits yet. */
	| "empty-repo"
	/** The listing exceeded what one recursive tree request returns. */
	| "truncated"
	/** No response: offline, DNS, timeout. */
	| "network"
	| "http";

export class GitHubError extends Error {
	constructor(
		message: string,
		readonly code: GitHubErrorCode,
		readonly status?: number,
		/** Epoch ms a rate limit lifts, when GitHub said. */
		readonly resetAt?: number,
	) {
		super(message);
		this.name = "GitHubError";
	}
}

export interface GitHubRepoConfig {
	owner: string;
	repo: string;
	token: string;
	/** Injectable for tests. */
	fetch?: typeof fetch;
	apiBase?: string;
	userAgent?: string;
}

/** A branch tip: the commit and the tree it points at, and when it was committed (ISO) when GitHub said. */
export interface RepoHead {
	commit: string;
	tree: string;
	date?: string;
}

export interface RemoteEntry {
	sha: string;
	size: number;
}

/** One file to change. `content: null` deletes it. */
export interface PushChange {
	path: string;
	content: Uint8Array | null;
}

const API_VERSION = "2022-11-28";
const TIMEOUT_MS = 30_000;
/** The slowest link a transfer is allowed for: 1 Mbit/s, in bytes per millisecond. */
const MIN_BYTES_PER_MS = 125;
/** Blobs larger than this upload one at a time, so each has the whole link its timeout assumes. */
const LARGE_BLOB_BYTES = 8 * 1024 * 1024;
const TREE_CHUNK = 300;
const CONCURRENCY = 8;

/** A request's timeout: the base one, plus time to move `bytes` over the slowest link allowed.
 *  A fixed 30 s cut off large source files (a 57 MB PDF is ~76 MB as base64), failing every round. */
export function transferTimeoutMs(bytes: number): number {
	return TIMEOUT_MS + Math.ceil(Math.max(0, bytes) / MIN_BYTES_PER_MS);
}

/** How many bytes a blob of `size` takes on the wire as base64. */
const base64Length = (size: number) => Math.ceil(size / 3) * 4;
/** Written to an empty repo so it has a first commit the Git Data API can build on. */
const BOOTSTRAP_PATH = ".learnharness-sync";

/** Run `fn` over `items`, at most `limit` at once, keeping the order. */
async function mapLimit<T, R>(items: readonly T[], limit: number, fn: (item: T) => Promise<R>): Promise<R[]> {
	const out = new Array<R>(items.length);
	let next = 0;
	const worker = async () => {
		while (next < items.length) {
			const i = next++;
			out[i] = await fn(items[i] as T);
		}
	};
	await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker));
	return out;
}

/** The bytes as UTF-8 text when they round-trip exactly (and hold no NUL),
 *  so they can ride in a tree entry's `content`; null for binary. */
function asUtf8(bytes: Uint8Array): string | null {
	const buf = Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength);
	const text = buf.toString("utf8");
	if (text.includes("\0")) return null;
	return Buffer.from(text, "utf8").equals(buf) ? text : null;
}

/** What the sync engine needs of a repo (GitHubRepo, or a fake in tests). */
export type RemoteRepo = Pick<GitHubRepo, "info" | "head" | "tree" | "blob" | "blobs" | "push">;

const encodeBranch = (branch: string) => branch.split("/").map(encodeURIComponent).join("/");

export class GitHubRepo {
	private readonly fetchImpl: typeof fetch;
	private readonly base: string;
	private readonly repoPath: string;

	constructor(private readonly cfg: GitHubRepoConfig) {
		this.fetchImpl = cfg.fetch ?? fetch;
		this.base = cfg.apiBase ?? "https://api.github.com";
		this.repoPath = `/repos/${encodeURIComponent(cfg.owner)}/${encodeURIComponent(cfg.repo)}`;
	}

	/** `expectBytes`: the size of the answer when it is known to be large (a blob), so its timeout allows for it. */
	private async request(method: string, urlPath: string, body?: unknown, expectBytes = 0): Promise<unknown> {
		const payload = body === undefined ? undefined : JSON.stringify(body);
		let res: Response;
		try {
			res = await this.fetchImpl(`${this.base}${urlPath}`, {
				method,
				headers: {
					accept: "application/vnd.github+json",
					authorization: `Bearer ${this.cfg.token}`,
					"x-github-api-version": API_VERSION,
					"user-agent": this.cfg.userAgent ?? "learnharness-sync",
					...(body === undefined ? {} : { "content-type": "application/json" }),
				},
				body: payload,
				// The signal also bounds reading the answer's body (res.json() below).
				signal: AbortSignal.timeout(transferTimeoutMs((payload?.length ?? 0) + expectBytes)),
			});
		} catch (err) {
			throw new GitHubError(`could not reach GitHub: ${err instanceof Error ? err.message : String(err)}`, "network");
		}
		if (res.ok) return res.status === 204 ? null : await res.json();

		let message = res.statusText;
		try {
			const parsed = (await res.json()) as { message?: unknown };
			if (typeof parsed.message === "string") message = parsed.message;
		} catch {
			/* no JSON body */
		}
		const detail = `GitHub answered ${res.status}: ${message}`;
		if (res.status === 401) throw new GitHubError(detail, "auth", 401);
		if (res.status === 403 || res.status === 429) {
			const retryAfter = Number(res.headers.get("retry-after"));
			const reset = Number(res.headers.get("x-ratelimit-reset"));
			const exhausted =
				res.status === 429 ||
				res.headers.get("x-ratelimit-remaining") === "0" ||
				(Number.isFinite(retryAfter) && retryAfter > 0) ||
				/rate limit/i.test(message);
			if (exhausted) {
				const resetAt = retryAfter > 0 ? Date.now() + retryAfter * 1000 : reset > 0 ? reset * 1000 : undefined;
				throw new GitHubError(detail, "rate-limit", res.status, resetAt);
			}
			throw new GitHubError(detail, "auth", res.status);
		}
		if (res.status === 404) throw new GitHubError(detail, "not-found", 404);
		if (res.status === 409 && /empty/i.test(message)) throw new GitHubError(detail, "empty-repo", 409);
		if (res.status === 422 && /fast.?forward/i.test(message)) throw new GitHubError(detail, "conflict", 422);
		throw new GitHubError(detail, "http", res.status);
	}

	/** Reachability and access check: the repo's default branch, and whether it is private. */
	async info(): Promise<{ defaultBranch: string; private: boolean }> {
		const body = (await this.request("GET", this.repoPath)) as { default_branch?: string; private?: boolean };
		return { defaultBranch: body.default_branch ?? "main", private: body.private === true };
	}

	/** The branch tip, or null when the branch has no commits yet (an empty
	 *  repo). A wrong repo throws "not-found" from info(), which the caller
	 *  checks first; here a missing branch is the empty-repo case. */
	async head(branch: string): Promise<RepoHead | null> {
		try {
			const body = (await this.request("GET", `${this.repoPath}/branches/${encodeBranch(branch)}`)) as {
				commit?: { sha?: string; commit?: { tree?: { sha?: string }; committer?: { date?: string } } };
			};
			const commit = body.commit?.sha;
			const tree = body.commit?.commit?.tree?.sha;
			if (!commit || !tree) throw new GitHubError("GitHub returned a branch without a commit", "http");
			const date = body.commit?.commit?.committer?.date;
			return date ? { commit, tree, date } : { commit, tree };
		} catch (err) {
			if (err instanceof GitHubError && (err.code === "not-found" || err.code === "empty-repo")) return null;
			throw err;
		}
	}

	/** Every file in a tree, recursively: path → blob SHA and size. */
	async tree(treeSha: string): Promise<Record<string, RemoteEntry>> {
		const body = (await this.request("GET", `${this.repoPath}/git/trees/${treeSha}?recursive=1`)) as {
			tree?: { path: string; type: string; sha: string; size?: number }[];
			truncated?: boolean;
		};
		if (body.truncated) {
			throw new GitHubError("the repo has too many files to list in one request", "truncated");
		}
		const out: Record<string, RemoteEntry> = {};
		for (const e of body.tree ?? []) if (e.type === "blob") out[e.path] = { sha: e.sha, size: e.size ?? 0 };
		return out;
	}

	/** A file's bytes by blob SHA. `size` (bytes, from the tree listing) widens the timeout for a large file. */
	async blob(sha: string, size = 0): Promise<Uint8Array> {
		const body = (await this.request("GET", `${this.repoPath}/git/blobs/${sha}`, undefined, base64Length(size))) as {
			content?: string;
			encoding?: string;
		};
		if (body.encoding !== "base64" || typeof body.content !== "string") {
			throw new GitHubError("GitHub returned a blob in an unexpected encoding", "http");
		}
		return new Uint8Array(Buffer.from(body.content, "base64"));
	}

	/** Several blobs at once (bounded), keyed by SHA. `sizes` (SHA → bytes) widens each large one's timeout. */
	async blobs(shas: readonly string[], sizes: Readonly<Record<string, number>> = {}): Promise<Record<string, Uint8Array>> {
		const unique = [...new Set(shas)];
		const bytes = await mapLimit(unique, CONCURRENCY, (sha) => this.blob(sha, sizes[sha] ?? 0));
		return Object.fromEntries(unique.map((sha, i) => [sha, bytes[i] as Uint8Array]));
	}

	/** The first commit of an empty repo. The Git Data API refuses an empty
	 *  repository (409), so one marker file goes in through the contents API. */
	private async bootstrap(branch: string): Promise<RepoHead> {
		const body = (await this.request("PUT", `${this.repoPath}/contents/${BOOTSTRAP_PATH}`, {
			message: "Start LearnHarness sync",
			content: Buffer.from("LearnHarness sync repository. Managed by the app; edit nothing by hand.\n").toString("base64"),
			branch,
		})) as { commit?: { sha?: string; tree?: { sha?: string } } };
		const commit = body.commit?.sha;
		const tree = body.commit?.tree?.sha;
		if (!commit || !tree) throw new GitHubError("GitHub did not return the first commit", "http");
		return { commit, tree };
	}

	/**
	 * Push `changes` as ONE commit on `branch`, on top of `parent` (the tip
	 * the caller read; null for an empty repo). Fails with "conflict" when the
	 * branch moved in the meantime (never forces): the caller pulls and retries.
	 */
	async push(branch: string, parent: RepoHead | null, changes: readonly PushChange[], message: string): Promise<RepoHead> {
		const base = parent ?? (await this.bootstrap(branch));
		if (changes.length === 0) return base;

		const toEntry = async (c: PushChange) => {
			const entry = { path: c.path, mode: "100644", type: "blob" } as const;
			if (c.content === null) return { ...entry, sha: null };
			const text = asUtf8(c.content);
			if (text !== null) return { ...entry, content: text };
			const blob = (await this.request("POST", `${this.repoPath}/git/blobs`, {
				content: Buffer.from(c.content).toString("base64"),
				encoding: "base64",
			})) as { sha: string };
			return { ...entry, sha: blob.sha };
		};
		// Small files go CONCURRENCY at a time; large ones one at a time, so a slow link is not split eight ways.
		const isLarge = (c: PushChange) => c.content !== null && c.content.byteLength > LARGE_BLOB_BYTES;
		const indexes = changes.map((_, i) => i);
		const entries: Awaited<ReturnType<typeof toEntry>>[] = new Array(changes.length);
		for (const [group, limit] of [
			[indexes.filter((i) => !isLarge(changes[i] as PushChange)), CONCURRENCY],
			[indexes.filter((i) => isLarge(changes[i] as PushChange)), 1],
		] as const) {
			const made = await mapLimit(group, limit, (i) => toEntry(changes[i] as PushChange));
			group.forEach((i, k) => (entries[i] = made[k] as Awaited<ReturnType<typeof toEntry>>));
		}

		let tree = base.tree;
		for (let i = 0; i < entries.length; i += TREE_CHUNK) {
			const made = (await this.request("POST", `${this.repoPath}/git/trees`, {
				base_tree: tree,
				tree: entries.slice(i, i + TREE_CHUNK),
			})) as { sha: string };
			tree = made.sha;
		}
		const made = (await this.request("POST", `${this.repoPath}/git/commits`, {
			message,
			tree,
			parents: [base.commit],
		})) as { sha: string };
		await this.request("PATCH", `${this.repoPath}/git/refs/heads/${encodeBranch(branch)}`, { sha: made.sha, force: false });
		return { commit: made.sha, tree };
	}
}
