from memory.decision import query_text_phase_2

if __name__ == "__main__":
    print("Enter your query:")

    while True:
        user_query = input(">> ")
        if(user_query.lower() in ['exit', 'quit','close']):
            print("Exiting...")
            break

        result = query_text_phase_2(user_query)

        if not result or not result.get('threads'):
            print("No relevant messages found.")
            continue
        
        if result['is_fallback']:
            print(f"{result['fallback_reason']}. Showing older relevant results instead.")
        
        if result['type'] == 'broad':
            print(f"Broad query — found {len(result['threads'])} relevant threads:\n")
        elif result['type'] == 'ambiguous':
            print(f"Ambiguous query — found {len(result['threads'])} possibly relevant threads:\n")
        else:
            print(f"\nFound 1 relevant thread:\n")

        for i, thread in enumerate(result['threads']):
            if len(result['threads']) > 1:
                print(f"--- Thread {i+1} (id: {thread['thread_id']}) ---")
            for item in thread['messages']:
                meta = item['metadata']
                text = item['document']
                print(f"  @{meta['user']} [{meta['ts']}]: {text}")
            print()
            