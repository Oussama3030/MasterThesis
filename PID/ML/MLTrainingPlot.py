
# Load and prepare data
training_samples = np.load('training_samples.npy')
mapped_targets = np.load('mapped_targets.npy')

# Apply a P range

# pt_range = (0.5, 1.0)
# mask = (training_samples[:,1] > 0.75) 

# training_samples = training_samples[mask]
# mapped_targets = mapped_targets[mask]

# Create DataFrame
features = ["dE/dx", "P", "tofBeta"]
            

            # , "tpcNSigmaEl", "tofNSigmaEl", 
            #      "tofNSigmaPr", "tpcNSigmaPr", "tofNSigmaPi", "tpcNSigmaPi", 
            #     "tofNSigmaKa", "tpcNSigmaKa"]

# Split the data
training_input, testing_input, training_target, testing_target = train_test_split(training_samples, 
                                                    mapped_targets, 
                                                    test_size=0.3, # 20% test, 80% train
                                                    stratify=mapped_targets, # Ensures similar ratio's in test and training 
                                                    random_state=42) # make the random split reproducible
