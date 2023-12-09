        self.test_datasize = test_datasize
        
        for i in range(0,self.test_datasize):
            filename = "test/test"+str(i)+".csv"
            testdata = pd.read_csv(filename)
            test_full_data = testdata.values
            if i == 0:
                self.test_input = np.expand_dims(test_full_data[:-1],axis = 0) # delete last data
                self.test_output = np.expand_dims(test_full_data[:,0:2],axis = 0)
                test_input_seq, test_output_seq = self.seq_data(self.test_input[i], self.test_output[i], sequence_length)
                self.test_input_seq = test_input_seq.unsqueeze(0)
                self.test_output_seq = test_output_seq.unsqueeze(0)

            else:
                self.test_input = np.append(self.test_input, np.expand_dims(test_full_data[:-1],axis = 0),axis = 0) # delete last data
                self.test_output = np.append(self.test_output,  np.expand_dims(test_full_data[:,0:2],axis = 0),axis = 0)
                test_input_seq, test_output_seq = self.seq_data(self.test_input[i], self.test_output[i], sequence_length)
                self.test_input_seq = torch.cat((self.test_input_seq, test_input_seq.unsqueeze(0)))
                self.test_output_seq = torch.cat((self.test_output_seq, test_output_seq.unsqueeze(0)))