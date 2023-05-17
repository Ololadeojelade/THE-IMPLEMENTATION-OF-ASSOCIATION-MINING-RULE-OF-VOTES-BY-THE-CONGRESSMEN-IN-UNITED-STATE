# THE IMPLEMENTATION OF ASSOCIATION MINING RULE OF VOTES BY THE CONGRESSMEN IN UNITED STATE
Machine Learning using Association rule mining to predict votes by Congress Men

### INTRODUCTION

Association rules mining is a technique to uncover how items are associated to each other. It uses algorithms such as Apriori are very useful for finding simple associations between our data items. An association rule has two parts: an antecedent and a consequent. An antecedent is an item found within the data while a consequent is an item found in combination with the antecedent. Association rules are created by searching data for frequent if-then patterns and using the criteria support and confidence to identify the most important relationships.
There are three main measures used in association rule or mining. These measures are support, confidence, and lift. 

The support measure is based on the percentage of transactions that an item set appears within a given dataset. Support is the default popularity of an item or a variable within a dataset. This is usually determined by dividing the total number of transactions by the number of transactions that contain a certain item or variable. Support measure also refers to the percentage of variables on the left- and right-hand side are present.

The confidence measure simply means that number of times an item is likely to be purchase when another item is purchased. In the case of this course work the confidence value would be measured by the number of votes of a particular variable gets when another variable is also voted for. For example, if the voters agree on voting for Physician-fee-freeze there is a likelihood that they would also vote for Duty-free-exports. The confidence measure also refers to the percentage of variables on the left-hand that also contained variables on the right hand.

The lift measure is basically the reverse of the confidence measure. For example, what this means is that when voters vote in favour of Duty-free-exports there is a probability that the voters would also vote in favour of Physician-fee-freeze. The lift measure measures how much more frequently the left-hand variables are found in the right hand than without the right hand.
Using the apriori class by sending the transactions object and identifying the minimum support and confidence.


## EXPLANATION AND PREPARATION OF DATASET
This data set includes votes for each of the U.S. House of Representatives Congressmen on the 16 key votes identified by the CQA.

![image](https://github.com/Orlawlardey/THE-IMPLEMENTATION-OF-ASSOCIATION-MINING-RULE-OF-VOTES-BY-THE-CONGRESSMEN-IN-UNITED-STATE/assets/124607057/9f028313-1c19-4548-afb8-c35dca9f9a95)


There were empty cells as ? in the exported dataset. The ? were changed to NA and after the file was imported into python the NA values were removed using this code House_data=House_data.dropna(axis=0). The NA values were removed from the entire dataset to get the actual figure or responses on the dataset being used.

## IMPLEMENTATION IN PYTHON

The necessary libraries were imported into python for the association mining task.

![image](https://github.com/Orlawlardey/THE-IMPLEMENTATION-OF-ASSOCIATION-MINING-RULE-OF-VOTES-BY-THE-CONGRESSMEN-IN-UNITED-STATE/assets/124607057/6a901739-f08e-48c2-a531-c447f114e7a2)

After importing the important libraries, the dataset file was imported into python. From the dataset we can see that we have two parties of voters the Republicans and the Democrats. We also have a total count of 233 responses which falls under the Yes or No categories.

![image](https://github.com/Orlawlardey/THE-IMPLEMENTATION-OF-ASSOCIATION-MINING-RULE-OF-VOTES-BY-THE-CONGRESSMEN-IN-UNITED-STATE/assets/124607057/57202393-dc30-42e6-8838-a531b91aa867)

![image](https://github.com/Orlawlardey/THE-IMPLEMENTATION-OF-ASSOCIATION-MINING-RULE-OF-VOTES-BY-THE-CONGRESSMEN-IN-UNITED-STATE/assets/124607057/5e0dad0c-d2f9-4152-960d-0a0cd535b534)

To view the number of responses for each column a bar plot was created and can be seen below.

![image](https://github.com/Orlawlardey/THE-IMPLEMENTATION-OF-ASSOCIATION-MINING-RULE-OF-VOTES-BY-THE-CONGRESSMEN-IN-UNITED-STATE/assets/124607057/8471c04d-1c94-4e89-8cf9-2accc48e6ffa)


![image](https://github.com/Orlawlardey/THE-IMPLEMENTATION-OF-ASSOCIATION-MINING-RULE-OF-VOTES-BY-THE-CONGRESSMEN-IN-UNITED-STATE/assets/124607057/c7c8acae-d789-415f-868c-e003afd333d8)

Where Yes is blue, and No is orange from both parties we can see that Export-administration-act-south-Africa has more positive responses while Synfuels-corporation-cutback has more negative responses.
We would then create a transaction for this dataset using a few selected columns. This is important because a transaction is an operation that modifies the state of my database.

![image](https://github.com/Orlawlardey/THE-IMPLEMENTATION-OF-ASSOCIATION-MINING-RULE-OF-VOTES-BY-THE-CONGRESSMEN-IN-UNITED-STATE/assets/124607057/2eca56de-0694-4a7e-b607-c9495a507f4e)

Using utils in a transaction is important as it helps collect simple Python classes and functions that shortens and simplify common patterns.

![image](https://github.com/Orlawlardey/THE-IMPLEMENTATION-OF-ASSOCIATION-MINING-RULE-OF-VOTES-BY-THE-CONGRESSMEN-IN-UNITED-STATE/assets/124607057/d3dbe352-3dc7-48ee-8fa5-0d58cc06ce5a)


From the image above we would be working with 15 voters from the entire dataset. The minimum support value was placed at 0.5 because we are working with a large data file and to enable us focus on just a few numbers of responses gotten from the voters in each party we had ensure that the confidence value is higher than the support value. This also applies to the confidence value that was placed at 0.6.
The support value shows how different variables in the dataset support each other. The confidence value gives the assurance that the variables of the right-hand side would go alongside the variables on the left-hand side. The lift value helps us determine whether there are positive or negative correlations between other values.


![image](https://github.com/Orlawlardey/THE-IMPLEMENTATION-OF-ASSOCIATION-MINING-RULE-OF-VOTES-BY-THE-CONGRESSMEN-IN-UNITED-STATE/assets/124607057/3d3c9027-59ca-4e98-b520-c4907b6c30cf)

From the image above we can see that rows 1 and 2 have the same lift value of 46.83%. What this simply means is that the those who voted for the El Salvador-aid variable also voted for the crime variable even though those in the crime variable did not necessarily vote for the El Salvador-aid.

![image](https://github.com/Orlawlardey/THE-IMPLEMENTATION-OF-ASSOCIATION-MINING-RULE-OF-VOTES-BY-THE-CONGRESSMEN-IN-UNITED-STATE/assets/124607057/9990152c-dbf5-498e-82ca-2e62ce643b3f)

![image](https://github.com/Orlawlardey/THE-IMPLEMENTATION-OF-ASSOCIATION-MINING-RULE-OF-VOTES-BY-THE-CONGRESSMEN-IN-UNITED-STATE/assets/124607057/90a43147-57e3-4316-b39b-7adde08d2503)

From the image above we can see that the support values for the variables in the last three rows with confidence values 79%, 97% and 61% we can see that the variables on the left-hand side can also be found on the right-hand side. What this simply means is that those who voted for religious groups in schools also voted for El Salvador aid, those who voted for aid to Nicaraguan contras also voted for export administration act South Africa and those who voted for export administration act South Africa also voted for aid to Nicaraguan contras.

![image](https://github.com/Orlawlardey/THE-IMPLEMENTATION-OF-ASSOCIATION-MINING-RULE-OF-VOTES-BY-THE-CONGRESSMEN-IN-UNITED-STATE/assets/124607057/4471d6f0-c4fd-4634-a71b-94e382116b69)

![image](https://github.com/Orlawlardey/THE-IMPLEMENTATION-OF-ASSOCIATION-MINING-RULE-OF-VOTES-BY-THE-CONGRESSMEN-IN-UNITED-STATE/assets/124607057/704bfa11-a55a-436a-b242-264593e0bd20)


Looking at the result derived from the confidence values, we can see that the confidence values for many variables is very high. This simply means that many of the variables on the left-hand side can also be found on the right-hand side. For example, those who voted for the Adoption of the budget solution also voted for the export administration act South Africa.
Because the top rules have empty itemset in the support measure, we would be using the lambda function to remove the rules with empty list in the LHS column of the given dataset.

![image](https://github.com/Orlawlardey/THE-IMPLEMENTATION-OF-ASSOCIATION-MINING-RULE-OF-VOTES-BY-THE-CONGRESSMEN-IN-UNITED-STATE/assets/124607057/ed7aadce-b029-4b04-8161-df4126a36eaf)


![image](https://github.com/Orlawlardey/THE-IMPLEMENTATION-OF-ASSOCIATION-MINING-RULE-OF-VOTES-BY-THE-CONGRESSMEN-IN-UNITED-STATE/assets/124607057/1e57156e-aa98-49b3-9d3c-ddf2c0215655)


To review a lesser number of voters within the dataset the minimum support value remained the same while the confidence value was increased. This therefore reduced our voters’ size from 15 voters to 10 voters.

![image](https://github.com/Orlawlardey/THE-IMPLEMENTATION-OF-ASSOCIATION-MINING-RULE-OF-VOTES-BY-THE-CONGRESSMEN-IN-UNITED-STATE/assets/124607057/c3588c64-5f2a-40d7-9468-cd65c2adaa49)


![image](https://github.com/Orlawlardey/THE-IMPLEMENTATION-OF-ASSOCIATION-MINING-RULE-OF-VOTES-BY-THE-CONGRESSMEN-IN-UNITED-STATE/assets/124607057/dc425e60-f2aa-4340-82b4-c357107b2757)


![image](https://github.com/Orlawlardey/THE-IMPLEMENTATION-OF-ASSOCIATION-MINING-RULE-OF-VOTES-BY-THE-CONGRESSMEN-IN-UNITED-STATE/assets/124607057/2d58b4c5-b270-4be8-ab9d-682a4017b16c)


![image](https://github.com/Orlawlardey/THE-IMPLEMENTATION-OF-ASSOCIATION-MINING-RULE-OF-VOTES-BY-THE-CONGRESSMEN-IN-UNITED-STATE/assets/124607057/b68647cc-1dd6-43df-a92c-cf4eff303c3c)


Since the purpose of this report is to know the variable that had the most positive and negative feedback from the entire dataset, we would therefore plot a chart to view this.

![image](https://github.com/Orlawlardey/THE-IMPLEMENTATION-OF-ASSOCIATION-MINING-RULE-OF-VOTES-BY-THE-CONGRESSMEN-IN-UNITED-STATE/assets/124607057/1a465801-6771-4099-a6d1-fc28377b236c)

From the chart above, we can see that Export administration act South Africa had the highest number of positive responses while Synfuels corporation cutback had the highest number of negative responses.

![image](https://github.com/Orlawlardey/THE-IMPLEMENTATION-OF-ASSOCIATION-MINING-RULE-OF-VOTES-BY-THE-CONGRESSMEN-IN-UNITED-STATE/assets/124607057/ba56190d-7b43-489c-a0c6-67014089315a)


To get the support, confidence and lift for the Export administration act South Africa we set our minimum support to 0.5 and our minimum confidence level to 0.7. The result of this gives us 3 associated rules which can be seen in the image above.

![image](https://github.com/Orlawlardey/THE-IMPLEMENTATION-OF-ASSOCIATION-MINING-RULE-OF-VOTES-BY-THE-CONGRESSMEN-IN-UNITED-STATE/assets/124607057/c814575e-3b1b-448a-af17-1cc842d3e8fb)


![image](https://github.com/Orlawlardey/THE-IMPLEMENTATION-OF-ASSOCIATION-MINING-RULE-OF-VOTES-BY-THE-CONGRESSMEN-IN-UNITED-STATE/assets/124607057/a04ffe9b-7745-4c17-9922-45467fbdcc82)


## RESULT ANALYSIS AND DISCUSSION

The result from this analysis shows that different variables have different support, confidence and lift level. It can be said that the greater the lift of the concerned variable the stronger the rule for the variable.

## CONCLUSION 
We can see that the minimum support and minimum confidence level used within a data set determine the size of the data set that would be worked on within the data set.



