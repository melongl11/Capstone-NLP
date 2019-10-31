import ijson


def load_json(filename):
    n = 0
    doc_count = 0
    output_name = "namuwiki_data"

    with open(filename, 'r') as fd:
        parser = ijson.parse(fd)
        write_file = open('dataset/' + output_name + str(doc_count) + '.txt', 'w', encoding='utf8')
        for prefix, event, value in parser:
            if prefix.endswith('.text'):
                write_file.write(value)
            n = n + 1
            if n % 5000 == 0:
                print("%d 번째 While 문" % n)
            if n % 1000000 == 0:
                write_file.close()
                doc_count += 1
                write_file = open('dataset/' + output_name + str(doc_count) + '.txt', 'w', encoding='utf8')
        write_file.close()



if __name__ == "__main__":
    load_json('dataset/namuwiki_20190312.json')
