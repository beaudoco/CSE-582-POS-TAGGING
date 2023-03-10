#After testing, show results of predicted WE with actual
def test_predicted_list(words, predicted): #len= 832
    compare_list=[]

    for i in range(len(words)):
        compare_list.append([words[i], predicted[i]])


    #Write to text file
    with open(r'./output.txt', 'w') as fp:
        for item in compare_list:
            # write each item on a new line
            fp.write("%s\n" % ' '.join(item))
