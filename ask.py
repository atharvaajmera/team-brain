from memory.decision import query_text_phase_2
from memory.llm import generate_response

def _tid_label(thread_id):
    """Convert thread_id like 1773000000.0 to a short label."""
    return f"thread:{str(int(float(thread_id)))}"

if __name__ == "__main__":
    print("Enter your query:")

    while True:
        user_query = input(">> ")
        if user_query.lower() in ['exit', 'quit', 'close']:
            print("Exiting...")
            break

        result = query_text_phase_2(user_query)

        if not result or not result.get('threads'):
            print("No relevant messages found.")
            continue

        if result['is_fallback']:
            print(f"  (fallback: {result['fallback_reason']})")

        category = result['type'].upper()
        stats = result.get('stats', {})

        # -- Category & retrieval stats --
        print(f"\n  Category : {category}")
        if stats:
            print(f"  Stats    : signal_norm={stats.get('signal_norm','?')}  "
                  f"rel_gap={stats.get('rel_gap','?')}  "
                  f"entropy={stats.get('entropy','?')}  "
                  f"threads={stats.get('n_threads','?')}")

        # -- LLM response --
        answer = generate_response(user_query, category, result['threads'])
        print(f"\n  {answer}")

        # -- Citations --
        thread_ids = [t['thread_id'] for t in result['threads']]
        citations = ", ".join(_tid_label(tid) for tid in thread_ids)
        print(f"\n  Sources  : [{citations}]\n")
            