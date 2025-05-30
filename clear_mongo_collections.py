from mongo_utils import delete_all_vcons, delete_all_faqs

def main():
    delete_all_vcons()
    delete_all_faqs()

if __name__ == "__main__":
    main()