from memory import query_text_phase_2

if __name__ == "__main__":
    print("Enter your query:")

    while True:
        user_query = input(">> ")
        if(user_query.lower() in ['exit', 'quit','close']):
            print("Exiting...")
            break

        res=query_text_phase_2(user_query)

        if not res:
            print("No relevant messages found.")
            continue
        print("Top relevant messages:")
        for item in res:
            meta=item['metadata']
            text=item['document']
            # confidence=item['distance']
            print(f"User ({meta['user']}) at {meta['ts']}: said {text}")
            