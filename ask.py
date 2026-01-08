from memory import query_text

threshold= 1.2

if __name__ == "__main__":
    print("Enter your query:")

    while True:
        user_query = input(">> ")
        if(user_query.lower() in ['exit', 'quit','close']):
            print("Exiting...")
            break

        res=query_text(user_query)

        if not res:
            print("No relevant messages found.")
            continue
        print("Top relevant messages:")
        for item in res:
            meta=item['metadata']
            text=item['document']
            confidence=item['distance']
            if confidence <= threshold:
                 print(f"User ({meta['user']}) at {meta['ts']}: said {text} with a confidence score of {confidence}\n")
            else:
                 print(f"Skipping message from User ({meta['user']}) at {meta['ts']} due to low confidence score of {confidence}\n")
