## Data Collection

------------------


1. Collect all the coverage and faulty lines csv files
2. Split them into two categories
   1. Train dataset (80% of all files)
   2. Test dataset (20% of all files)
   3. Set the same random seed for reproducibility
3. Obtain size of the coverage training dataset (`data_size`)

## Training

------------------

1. Iterate in range of `data_size`
2. Create FaultLocalizator object
   1. To initialize the object we need to pass coverage file, to compare the results we can also additionally add ground truth (faulty lines) and set a flag whether an object is a benchmark model or not
   2. Run hand-made parser (`faults_extractor.py`) to parse CSV file to extract faults.
   3. Check how many faults from faulty lines csv file are covered by coverage file
   4. If zero -> running the code further is pointless (our solution cannot handle faults of omittance)
   5. Check the flag of the object
      1. If object is benchark, we need to calculate Jaccard coefficient
      2. Otherwise, we calculate other formulas
         - D star 
         - D star 3 
         - Tarantula 
         - Ochiai
         - Naish
         - GP26
3. Iterate on all formulas which object has computed
   1. Calculate ranking based on formula
   2. Calculate normalized exam score
   3. Calculate top N faults
   4. Check how many real faults were found within the top

4. (*) Create benchmark object, repeat steps 2 and 3
5. (*) Calculate Wilcoxon test on training
6. (*) Plot exam scores and top N accuracy of formulas
7. Calculate weights of each formula

## Testing

------------------

1. Repeat steps 1 and 2
2. Based on weights from training and scores from testing calculate majority voting score
3. Calculate the exam score of proposed method
4. (*) Create benchmark object and like in testing repeat steps
5. (*) Calculate Wilcoxon test on testing
6. (*) Plot exam score of Jaccard and proposed method

(*) - Not necessary for the algorithm