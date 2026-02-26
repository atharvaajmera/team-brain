from memory.decision import query_text_phase_2

if __name__ == "__main__":
    print("Enter your query:")

    while True:
        user_query = input(">> ")
        if(user_query.lower() in ['exit', 'quit','close']):
            print("Exiting...")
            break

        result = query_text_phase_2(user_query)

        if not result or not result.get('messages'):
            print("No relevant messages found.")
            continue
        
        if result['is_fallback']:
            print(f"{result['fallback_reason']}. Showing older relevant results instead.")
        
        print("Top relevant messages:")
        for item in result['messages']:
            meta=item['metadata']
            text=item['document']
            # confidence=item['distance']
            print(f"User ({meta['user']}) at {meta['ts']}: said {text}")
            