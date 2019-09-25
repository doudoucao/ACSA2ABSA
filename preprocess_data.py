# data combine
def combine_acsa_absa(acsa_path, absa_path):
    acsa_absa = open('./data/combine/acsa_absa_train.txt','a')
    with open(acsa_path,encoding='utf-8') as f:
        for line in f:
            result = line.split('\t')
            if result is not None:
                text = result[1].lower().strip()
                aspect = ' '.join(result[2].lower().strip().split('_'))
                polarity = result[3].strip()
                task = str(0)
                acsa_absa.write(task+ '\t' + text + '\t' + aspect + '\t' + polarity + '\n')

    absa_data = open(absa_path, 'r')
    lines = absa_data.readlines()
    absa_data.close()

    for i in range(0, len(lines), 3):
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].lower().strip()
        polarity = lines[i + 2].strip()
        text = text_left + " " + aspect + " " + text_right
        task = str(1)
        acsa_absa.write(task+ '\t' + text + '\t' + aspect + '\t' + polarity + '\n')

    acsa_absa.close()

def format_testdata(path):
    absa_test_data = open('./data/combine/absa_testdata.txt', 'a')
    absa_data = open(path, 'r')
    lines = absa_data.readlines()
    absa_data.close()

    for i in range(0, len(lines), 3):
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].lower().strip()
        polarity = lines[i + 2].strip()
        text = text_left + " " + aspect + " " + text_right
        task = str(1)
        absa_test_data.write(task + '\t' + text + '\t' + aspect + '\t' + polarity + '\n')

    absa_test_data.close()
# combine_acsa_absa('/media/sihui/0000339400042A0A/czx/ACSA2ABSA/data/MGAN/data/restaurant/train.txt',
#                   '/media/sihui/000970CB000A4CA8/Sentiment-Analysis/data/semeval14/Restaurants_Train.xml.seg')
format_testdata('/media/sihui/000970CB000A4CA8/Sentiment-Analysis/data/semeval14/Restaurants_Test_Gold.xml.seg')
