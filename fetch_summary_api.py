import json
from urllib.request import urlopen
while True:
# create getting started variables
    api = "https://www.googleapis.com/books/v1/volumes?q=isbn:"
    isbn = input("Enter 10 digit ISBN: ").strip()
# send a request and get a JSON response
    resp = urlopen(api + isbn)
# parse JSON into Python as a dictionary
    book_data = json.load(resp)
# create additional variables for easy querying
    volume_info = book_data["items"][0]["volumeInfo"]
    author = volume_info["authors"]
# practice with conditional expressions!
    prettify_author = author if len(author) > 1 else author[0]
# display title, author, page count, publication date
    print(f"\nTitle: {volume_info['title']}")
    print(f"Author: {prettify_author}")
    print(f"Page Count: {volume_info['pageCount']}")
    print(f"Description: {volume_info['description']}")
    print(f"Publication Date: {volume_info['publishedDate']}")
    print("\n***\n")
# ask user if they would like to enter another isbn
    user_update = input("Would you like to enter another ISBN? y or n ").lower().strip()
    if user_update != "y":
        print("May the Zen of Python be with you. Have a nice day!")
        break # as the name suggests, the break statement breaks out of the while loop
