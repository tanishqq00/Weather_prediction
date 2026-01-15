from splitting import split_data

def input_target():
    
    train_df, val_df, test_df = split_data()
    
    train_inputs = train_df.drop(columns=['Date','RainTomorrow'])
    train_target = train_df['RainTomorrow']
    
    val_inputs = val_df.drop(columns=['Date','RainTomorrow'])
    val_target = val_df['RainTomorrow']
    
    test_inputs = test_df.drop(columns=['Date','RainTomorrow'])
    test_target = test_df['RainTomorrow']
    
    return train_inputs, train_target, val_inputs, val_target, test_inputs, test_target
