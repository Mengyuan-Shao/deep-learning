import split_folders

path = "/Users/shaomengyuan/flowers"
out_path = "/Users/shaomengyuan/assignment"

# split_folders.fixed(input = path, output= out_path, seed= 1, fixed=(.6, .2, .2), oversample= False)

# split the whole dataset to train, test and validate where 60% of them are used to train,
# 20% of them are used to test and the rest 20% are used to validate.
split_folders.ratio(input = path, 
                    output= out_path, 
                    seed= 2, 
                    ratio= (.6, .2, .2))