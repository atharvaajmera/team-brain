"""
Synthetic Slack conversation generator.

Simulates:
  - 5-10 users with distinct personalities
  - Threaded conversations on realistic dev-team topics
  - Message bursts (quick back-and-forth)
  - Random delays between messages
  - Occasional long messages (explanations, stack traces, code snippets)
  - Slack API rate-limit-safe delays when seeding directly

Usage:
    python slack_convos_generator.py              # preview only (dry run)
    python slack_convos_generator.py --seed       # generate + seed into ChromaDB
    python slack_convos_generator.py --seed --clear  # clear DB first then seed
"""

import random
import time
import sys
import json
import os
import ssl
import certifi
from datetime import datetime
from dotenv import load_dotenv
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

load_dotenv()

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_CHANNEL_ID = os.getenv("SLACK_CHANNEL_ID_AUTO")  # e.g. C08XXXXXXXX

ssl_context = ssl.create_default_context(cafile=certifi.where())
client = WebClient(token=SLACK_BOT_TOKEN, ssl=ssl_context)

# ─── Users ─────────────────────────────────────────────────────────────────
USERS = [
    "alice",     # backend lead
    "bob",       # QA / testing
    "charlie",   # devops
    "diana",     # frontend lead
    "eve",       # fullstack
    "frank",     # data / DB
    "grace",     # security
    "henry",     # mobile
    "irene",     # PM / product
    "jake",      # junior dev
]

# ─── Topic templates ────────────────────────────────────────────────────────
# Each topic is a list of message templates.
# {user} is filled in dynamically.
# Some slots hold multiple alternatives — one is picked at random.

TOPIC_TEMPLATES = [
    # ── Auth / OAuth ──────────────────────────────────────────────────────
    {
        "topic": "OAuth token refresh failing",
        "owner": "alice",
        "messages": [
            ("alice",  "OAuth token refresh is returning 401 in staging. Anyone seeing this?",                                        "short"),
            ("bob",    "Confirmed on my end too. Seems like the refresh grant is being rejected.",                                     "short"),
            ("alice",  "The client_secret in the .env might be stale — last rotated 3 months ago.",                                  "short"),
            ("frank",  "I rotated the secret in Vault yesterday, but didn't push the env update to staging.",                         "short"),
            ("alice",  "That explains it. Updating staging .env now.",                                                                 "short"),
            ("alice",  "Fixed. Token refresh working again. I'll add a reminder to rotate secrets quarterly.",                        "medium"),
        ]
    },
    {
        "topic": "Google login redirect loop",
        "owner": "eve",
        "messages": [
            ("eve",    "Getting an infinite redirect loop on the Google login flow in prod.",                                          "short"),
            ("alice",  "Is the GOOGLE_CALLBACK_URL still pointing to localhost?",                                                     "short"),
            ("eve",    "Oh no. Yes it is. Missed that in the deployment checklist.",                                                  "short"),
            ("charlie","Updated the env var. Redeploying now.",                                                                       "short"),
            ("eve",    "Login flow working. I'll add a callback URL check to our smoke tests.",                                       "short"),
        ]
    },

    # ── Database ──────────────────────────────────────────────────────────
    {
        "topic": "Slow queries on orders table",
        "owner": "frank",
        "messages": [
            ("frank",  "The /api/orders endpoint is taking 4–6 seconds on large accounts. Running EXPLAIN now.", "short"),
            ("frank",  "Found it. A full table scan on `created_at` — no index.",                               "short"),
            ("alice",  "That table has 8 million rows. That scan is brutal.",                                   "short"),
            ("frank",  [
                "Creating the index now:\n```sql\nCREATE INDEX CONCURRENTLY idx_orders_created_at ON orders(created_at);\n```\nShould take ~10 minutes without locking.",
                "Adding composite index on (account_id, created_at) since most queries filter by account first."
            ],                                                                                                   "long"),
            ("bob",    "P99 latency dropped from 5.8s to 180ms after index. Confirmed in Datadog.",             "short"),
            ("frank",  "Adding index creation to the DB migration checklist.",                                   "short"),
        ]
    },
    {
        "topic": "DB migration rollback after failed deploy",
        "owner": "charlie",
        "messages": [
            ("charlie","Deploy to prod failed mid-migration. Users table is now in an inconsistent state.",     "short"),
            ("frank",  "Which migration number?",                                                               "short"),
            ("charlie","20240318_add_user_preferences. It added 3 columns but the FK constraint failed.",       "short"),
            ("frank",  "Rolling back: `flask db downgrade 20240317_baseline`",                                  "medium"),
            ("charlie","Rollback succeeded. Prod is stable again.",                                             "short"),
            ("irene",  "Do we need to communicate downtime to users? It was about 4 minutes.",                  "short"),
            ("alice",  "Sending status page update now.",                                                       "short"),
        ]
    },

    # ── CI / CD ───────────────────────────────────────────────────────────
    {
        "topic": "GitHub Actions build timeout",
        "owner": "charlie",
        "messages": [
            ("charlie","Main branch builds are timing out at the integration test step. Last 3 builds failed.", "short"),
            ("jake",   "Is it a flaky test or actually slow?",                                                  "short"),
            ("charlie","Slow. The test suite went from 4 min to 22 min after last merge.",                      "short"),
            ("alice",  "PR #441 added 80 new integration tests. No parallelism set.",                           "short"),
            ("charlie",[
                "Added `--parallel 4` to the pytest call and split the test suite across 3 runners in the matrix:\n```yaml\nstrategy:\n  matrix:\n    partition: [0, 1, 2]\n```",
                "Split tests by module across 3 parallel jobs. Each now runs in under 7 minutes."
            ],                                                                                                   "long"),
            ("bob",    "Green across all 3 partitions. Total pipeline time: 8 min.",                            "short"),
        ]
    },
    {
        "topic": "Docker image size ballooning",
        "owner": "diana",
        "messages": [
            ("diana",  "Our frontend Docker image grew from 280 MB to 1.1 GB after adding charting library.",   "short"),
            ("charlie","Multi-stage build? Are we including node_modules in the final stage?",                   "short"),
            ("diana",  "Yes — the COPY step is copying the whole project including dev dependencies.",           "short"),
            ("charlie",[
                "Fix is to use multi-stage build properly:\n```dockerfile\nFROM node:20 AS builder\nRUN npm ci --omit=dev\n\nFROM node:20-alpine\nCOPY --from=builder /app/dist ./dist\n```",
            ],                                                                                                   "long"),
            ("diana",  "Image is 310 MB now. Back to normal.",                                                  "short"),
        ]
    },

    # ── Frontend ──────────────────────────────────────────────────────────
    {
        "topic": "React hydration mismatch errors",
        "owner": "diana",
        "messages": [
            ("diana",  "Getting hydration mismatch errors on the product detail page in production only.",      "short"),
            ("jake",   "Is it SSR-related? I saw similar issues with dynamic content.",                         "short"),
            ("diana",  "Yes — the price component renders differently on server vs client because of currency formatting locale.",  "medium"),
            ("henry",  "We had the same on mobile web. `Intl.NumberFormat` defaults differ by environment.",    "short"),
            ("diana",  "Fixed by explicitly passing `locale='en-US'` to the formatter. Hydration errors gone.", "short"),
        ]
    },

    # ── Performance ───────────────────────────────────────────────────────
    {
        "topic": "Memory leak in background worker",
        "owner": "alice",
        "messages": [
            ("alice",  "Worker pod is getting OOM-killed every 6 hours. Memory grows linearly.",                "short"),
            ("charlie","K8s shows RSS at 1.8 GB before kill. Started after last week's deploy.",                "short"),
            ("alice",  [
                "Profiling with `memray`:\n```\nmemray run worker.py\nmemray flamegraph output.bin\n```\nFound a closure holding references to large DataFrame objects in the retry queue.",
            ],                                                                                                   "long"),
            ("frank",  "The retry queue was caching full response payloads instead of just IDs.",               "short"),
            ("alice",  "Fixed: cache only the message ID and re-fetch on retry. Memory stable at 320 MB.",      "short"),
            ("irene",  "Adding memory limit alerts at 80% threshold so we catch this earlier next time.",        "short"),
        ]
    },

    # ── Security ──────────────────────────────────────────────────────────
    {
        "topic": "Dependency vulnerability in lodash",
        "owner": "grace",
        "messages": [
            ("grace",  "Dependabot flagged CVE-2024-XXXX in lodash 4.17.20 — prototype pollution. High severity.", "medium"),
            ("diana",  "lodash is used in 14 packages in our frontend bundle.",                                   "short"),
            ("grace",  "The fix is upgrading to 4.17.21 or replacing with lodash-es for tree shaking.",          "short"),
            ("jake",   "Running `npm audit fix` — 11 of 14 auto-fixed.",                                         "short"),
            ("diana",  "The other 3 need manual resolution — they pin an older version.",                        "short"),
            ("grace",  [
                "Opened PRs for the 3 manual fixes:\n- PR #502: update chart-utils\n- PR #503: update form-validator\n- PR #504: update date-range-picker\nAll tested, no regressions.",
            ],                                                                                                    "long"),
            ("grace",  "All merged. Vulnerability resolved across the board.",                                    "short"),
        ]
    },
    {
        "topic": "Exposed API keys in git history",
        "owner": "grace",
        "messages": [
            ("grace",  "ALERT: An AWS API key was committed to the repo in commit a3f99bx. Rotating immediately.", "short"),
            ("alice",  "Which key? S3 or the SES sender?",                                                        "short"),
            ("grace",  "SES sender key. Rotating in AWS console now.",                                            "short"),
            ("charlie","Purging from git history with `git filter-repo`. Force-pushing to all branches.",         "short"),
            ("grace",  "New key live. Old key revoked. History cleaned.",                                         "short"),
            ("irene",  "Adding pre-commit hook with `detect-secrets` to all repos as follow-up.",                 "medium"),
        ]
    },

    # ── Infra / DevOps ────────────────────────────────────────────────────
    {
        "topic": "Redis cache eviction causing slowdowns",
        "owner": "frank",
        "messages": [
            ("frank",  "Cache hit rate dropped from 94% to 31% overnight. Lots of evictions.",                   "short"),
            ("charlie","maxmemory policy is set to `allkeys-lru`. Maybe the cache is full?",                     "short"),
            ("frank",  "Used memory is at 99.6%. We're caching raw API responses — some are 2 MB each.",         "short"),
            ("alice",  "We should compress the large payloads and set a TTL on them.",                           "short"),
            ("frank",  "Added gzip compression + 5 min TTL on responses over 100 KB. Cache usage down to 67%.", "medium"),
            ("bob",    "Cache hit rate back to 89%. API response times normal.",                                  "short"),
        ]
    },
    {
        "topic": "Kubernetes pod crash loop",
        "owner": "charlie",
        "messages": [
            ("charlie","api-gateway pod is in CrashLoopBackOff on prod cluster. Logs show OOMKilled.",           "short"),
            ("alice",  "Memory limit is 256 Mi. After last week's feature, it needs more.",                      "short"),
            ("charlie","Increasing limit to 512 Mi and request to 256 Mi.",                                      "short"),
            ("charlie",[
                "Updated resource spec:\n```yaml\nresources:\n  requests:\n    memory: '256Mi'\n    cpu: '250m'\n  limits:\n    memory: '512Mi'\n    cpu: '500m'\n```",
            ],                                                                                                    "long"),
            ("charlie","Pod is stable. Uptime 10 minutes and counting.",                                         "short"),
        ]
    },

    # ── Team / Process ────────────────────────────────────────────────────
    {
        "topic": "Sprint planning for Q2",
        "owner": "irene",
        "messages": [
            ("irene",  "Sprint planning for Q2 kicks off Monday. Please add your estimates to Jira by EOD Friday.", "medium"),
            ("alice",  "Backend team estimates are in.",                                                           "short"),
            ("diana",  "Frontend estimates added. Flagged 2 tickets as blockers — needs API contract first.",     "short"),
            ("bob",    "QA estimates in. We need an extra day for regression if the auth flow changes.",          "short"),
            ("irene",  "Noted. Auth flow changes are scheduled for week 3 to give QA buffer.",                    "short"),
            ("henry",  "Mobile estimates also added.",                                                             "short"),
            ("irene",  "Thanks everyone. Capacity looks good — sprint scope is locked.",                          "short"),
        ]
    },
    {
        "topic": "On-call handoff issues",
        "owner": "charlie",
        "messages": [
            ("charlie","Handoff notes from last on-call were incomplete — 2 incidents had no runbook entries.",   "medium"),
            ("alice",  "Which incidents?",                                                                         "short"),
            ("charlie","The S3 timeout on Sunday and the search index rebuild on Monday.",                        "short"),
            ("grace",  "I'll write up the S3 runbook today.",                                                     "short"),
            ("frank",  "I'll document the search index rebuild procedure.",                                        "short"),
            ("irene",  "Adding 'runbook updated' as a required checklist item for incident close.",               "short"),
        ]
    },

    # ── Mobile ────────────────────────────────────────────────────────────
    {
        "topic": "Push notifications not delivered on iOS",
        "owner": "henry",
        "messages": [
            ("henry",  "iOS push notifications are failing silently for about 20% of users since yesterday.",    "short"),
            ("grace",  "Is the APNs certificate current?",                                                        "short"),
            ("henry",  "Certificate expires in 2 days. That's probably it — devices might be rejecting it.",     "short"),
            ("grace",  "Renewing now. New cert uploaded to our push service.",                                    "short"),
            ("henry",  "Delivery rate back to 98%. The 2% are users with notifications disabled at OS level.",   "short"),
        ]
    },
]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_ts(base_ts, offset_seconds):
    """Return a unix timestamp string."""
    return str(int(base_ts + offset_seconds))


def _pick(val):
    """If val is a list of alternatives, pick one at random; otherwise return val."""
    return random.choice(val) if isinstance(val, list) else val


def _random_delay(style="normal"):
    """
    Return a seconds offset simulating realistic inter-message delay.
    burst  : 10–90 sec  (active back-and-forth)
    normal : 60–600 sec (regular conversation)
    slow   : 600–3600 sec (async, cross-timezone)
    """
    if style == "burst":
        return random.randint(10, 90)
    elif style == "slow":
        return random.randint(600, 3600)
    else:
        return random.randint(60, 600)


def _burst_pattern(n_messages):
    """
    Decide delay style for each message in a thread.
    Simulates: slow start → burst in the middle → trails off.
    """
    styles = []
    for i in range(n_messages):
        if i == 0:
            styles.append("normal")
        elif i < n_messages * 0.7:
            styles.append(random.choice(["burst", "burst", "normal"]))
        else:
            styles.append(random.choice(["normal", "slow"]))
    return styles


def generate_threads(base_ts=1780000000):
    """
    Generate all threads from TOPIC_TEMPLATES.

    Returns a list of dicts:
      { thread_ts, messages: [{ user, text, ts, thread_ts }] }
    """
    threads = []
    current_ts = base_ts

    for template in TOPIC_TEMPLATES:
        msgs = template["messages"]
        thread_ts = current_ts
        styles = _burst_pattern(len(msgs))

        thread_messages = []
        msg_ts = thread_ts

        for i, (user, text_val, _length) in enumerate(msgs):
            text = _pick(text_val)
            thread_messages.append({
                "user": user,
                "text": text,
                "ts": _make_ts(msg_ts, 0),
                "thread_ts": str(thread_ts),
            })
            if i < len(msgs) - 1:
                msg_ts += _random_delay(styles[i + 1])

        threads.append({
            "thread_ts": str(thread_ts),
            "topic": template["topic"],
            "messages": thread_messages,
        })

        # Advance base timestamp for next thread: at least 2 hours gap
        current_ts = msg_ts + random.randint(7200, 28800)

    return threads


def preview(threads):
    """Print generated threads to stdout without seeding."""
    print(f"\n{'='*80}")
    print(f"  Generated {len(threads)} threads")
    print(f"{'='*80}\n")
    for t in threads:
        ts_pretty = datetime.fromtimestamp(float(t["thread_ts"])).strftime("%Y-%m-%d %H:%M")
        print(f"  [{ts_pretty}] #{t['topic']}  ({len(t['messages'])} messages)")
        for m in t["messages"]:
            preview_text = m["text"][:80].replace("\n", " ")
            ellipsis = "…" if len(m["text"]) > 80 else ""
            print(f"    @{m['user']:<10} {preview_text}{ellipsis}")
        print()


def post_to_slack(threads, channel_id=None, msg_delay=1.2, thread_delay=3.0):
    """
    Post generated threads to a Slack channel via chat.postMessage.

    Flow per thread:
      1. Post the first message as a root message -> get ts
      2. Post remaining messages as replies using thread_ts=root_ts

    Rate limits (Slack Tier 3: chat.postMessage ~50 req/min):
      msg_delay   : seconds between messages in a thread  (default 1.2s)
      thread_delay: extra pause between threads           (default 3.0s)
    """
    ch = channel_id or SLACK_CHANNEL_ID
    if not ch:
        raise ValueError("SLACK_CHANNEL_ID not set. Add it to .env or pass channel_id= explicitly.")
    if not SLACK_BOT_TOKEN:
        raise ValueError("SLACK_BOT_TOKEN not set in .env")

    total_msgs = sum(len(t["messages"]) for t in threads)
    print(f"\n  Posting {len(threads)} threads / {total_msgs} messages to #{ch}...")
    print(f"  Est. time: ~{int(total_msgs * msg_delay + len(threads) * thread_delay)}s\n")

    posted_threads = 0
    posted_msgs = 0

    for t in threads:
        messages = t["messages"]
        root_ts = None

        for i, m in enumerate(messages):
            try:
                msg_metadata = {
                    "event_type": "team_brain_message",
                    "event_payload": {
                        "author": m["user"],
                        "channel": ch,
                        "thread_ts": root_ts if i > 0 else None,
                    }
                }
                if i == 0:
                    # Root message — starts the thread
                    resp = client.chat_postMessage(
                        channel=ch,
                        text=m["text"],
                        metadata=msg_metadata,
                    )
                    root_ts = resp["ts"]
                    print(f"  [{t['topic'][:40]}] root posted (ts={root_ts})")
                else:
                    # Reply in thread
                    client.chat_postMessage(
                        channel=ch,
                        text=m["text"],
                        thread_ts=root_ts,
                        metadata=msg_metadata,
                    )

                posted_msgs += 1
                time.sleep(msg_delay)

            except SlackApiError as e:
                err = e.response.get("error", str(e))
                if err == "ratelimited":
                    retry_after = int(e.response.headers.get("Retry-After", 30))
                    print(f"  Rate limited — waiting {retry_after}s...")
                    time.sleep(retry_after)
                    # retry once
                    client.chat_postMessage(
                        channel=ch,
                        text=m["text"],
                        thread_ts=root_ts if i > 0 else None,
                        metadata=msg_metadata,
                    )
                else:
                    print(f"  SlackAPIError on msg {i} in '{t['topic']}': {err}")

        posted_threads += 1
        time.sleep(thread_delay)

    print(f"\n  Done. {posted_threads} threads / {posted_msgs} messages posted to Slack.")


def export_json(threads, path="generated_convos.json"):
    """Save generated threads to a JSON file (useful for inspection / replay)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(threads, f, indent=2, ensure_ascii=False)
    print(f"  Exported to {path}")


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(42)  # remove for non-deterministic generation

    threads = generate_threads(base_ts=1780000000)

    if "--post" in sys.argv:
        # Optional: override channel from CLI e.g. --channel C08XXXXXXXX
        ch = None
        if "--channel" in sys.argv:
            idx = sys.argv.index("--channel")
            ch = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None
        post_to_slack(threads, channel_id=ch)
    else:
        preview(threads)
        if "--export" in sys.argv:
            export_json(threads)
        else:
            print("  (dry run — use --post to send to Slack, --export to save JSON)\n")
