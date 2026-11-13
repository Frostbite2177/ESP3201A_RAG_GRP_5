from repository_reader import extract_chunks


def main():
    chunks = extract_chunks('sample_repo')
    for chunk in chunks:
        print('=' * 40)
        print('File:', chunk['file'])
        print('Name:', chunk['name'])
        print('Type:', chunk['type'])
        print('Lines:', chunk['start_line'], '-', chunk['end_line'])
        print(chunk['code'])
    print(f'\nExtracted {len(chunks)} code chunks.')


if __name__ == '__main__':
    main()
