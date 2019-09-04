import arff;# resolve arff file
import copy;#realize deep copy
from math import log;#calculate Entropy
from graphviz import Digraph;#plot the tree
dot = Digraph(comment="Decision Tree");
d = open("d:/ftest.arff");
raw_data = arff.load(d);#resolve file
descriptiveFeature_value = {};
for f in raw_data['attributes']:
    descriptiveFeature_value[f[0]]=f[1];#get descriptive feature value
descriptive_Feature =list(descriptiveFeature_value.keys());#get descriptive feature

def caculate_Entropy(data):#calculate Entropy
    item_num = len(data);
    levelCounts = {};
    for item in data:
        item_target_feature=item[-1];
        if item_target_feature not in levelCounts.keys():
            levelCounts[item_target_feature] = 0;
        levelCounts[item_target_feature]+=1;
    entropy = 0.0;
    for key in levelCounts:
        prob = float(levelCounts[key])/item_num;
        entropy-= prob*log(prob,2);
    return entropy;


def split_data(data,feature,feture_value):#split data by feature value
    sub_data = [];
    tempdata = copy.deepcopy(data);
    for item in tempdata:
        if item[feature] == feture_value:
            del item[feature];
            sub_data.append(item);
    return sub_data;

def caculate_Remainder(data,features,values):#calculate rem
    remainder = {};
    for descriptive in range(len(features)-1):
        remainder[features[descriptive]] = 0;
        for value in values[features[descriptive]]:
            tempdata = split_data(data,descriptive,value);
            remainder[features[descriptive]]+=float(len(tempdata))/len(data)*caculate_Entropy(tempdata);
    return remainder;

def caculate_InformationGain(data,features,values):#calculate Infomation Gain
    data_entropy = caculate_Entropy(data);
    rem = caculate_Remainder(data,features,values);
    for r in rem:
        rem[r] = -rem[r]+data_entropy;
    print(rem);
    return rem;

def is_same_targetFeature(data):#jugde whether has been grouped
    count = {};
    for item in data:
        if item[-1] not in count.keys():
            count[item[-1]]=0;
        count[item[-1]] += 1;

    return count;

shallow = -1;
def generate_dicisionTree(data,features,values,great_feature,shallow,value):#generate abstract tree
    if (len(is_same_targetFeature(data)) == 0):
        return ;
    shallow+=1;
    if (len(is_same_targetFeature(data))==1):
#        print (value+" is classfied");
#        print(shallow);
        dot.node(name=data[0][-1],label=data[0][-1]);
        dot.edge(great_feature,data[0][-1],value);

        print(data[0][-1]+" is point to " + great_feature+" condition is "+value);


    else:
#        print(value + " is NOT classfied");
        inf_g = caculate_InformationGain(data,features,values);
        great_ig = 0;
        great_index = 0;
        flag = -1;
        father = great_feature;
        for ig in inf_g:
            flag+=1;
            if inf_g[ig]>great_ig:
                great_ig = inf_g[ig];
                great_index = flag;
        great_feature = features[great_index];
#        print(shallow);
        dot.node(name=great_feature+" ("+value+")",label=great_feature+value);
        dot.node(name=father,label=father)
        dot.edge(father,great_feature+" ("+value+")",value)
        print(great_feature+" is point to " + father+" Condition is "+ value);

        for fv in values[features[great_index]]:
            temp_data = copy.deepcopy(data);
            temp_feature = copy.deepcopy(features);
            temp_values = copy.deepcopy(values);
            del temp_feature[great_index];
            del temp_values[features[great_index]];
            generate_dicisionTree(split_data(temp_data,great_index,fv),temp_feature,temp_values,great_feature+" ("+value+")",shallow,fv);




data = raw_data['data'];
#print(caculate_Entropy(data));
#print(split_data(data,1,"low"));
generate_dicisionTree(data,descriptive_Feature,descriptiveFeature_value,"root",shallow,"null");
dot.render('test-output/test-table.gv', view=True)

