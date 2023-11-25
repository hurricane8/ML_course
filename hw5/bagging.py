import numpy as np

class SimplifiedBaggingRegressor:
    def __init__(self, num_bags, oob=False):
        self.num_bags = num_bags
        self.oob = oob
        
    def _generate_splits(self, data: np.ndarray):
        '''
        Generate indices for every bag and store in self.indices_list list
        '''
        self.indices_list = []
        data_length = len(data)
        for bag in range(self.num_bags):
            indices = np.random.choice(data_length, size=data_length, replace=True)
            self.indices_list.append(indices)
        
    def fit(self, model_constructor, data, target):
        '''
        Fit model on every bag.
        Model constructor with no parameters (and with no ()) is passed to this function.
        
        example:
        
        bagging_regressor = SimplifiedBaggingRegressor(num_bags=10, oob=True)
        bagging_regressor.fit(LinearRegression, X, y)
        '''
        self.data = None
        self.target = None
        self._generate_splits(data)
        assert len(set(list(map(len, self.indices_list)))) == 1, 'All bags should be of the same length!'
        assert list(map(len, self.indices_list))[0] == len(data), 'All bags should contain `len(data)` number of elements!'
        self.models_list = []
        for bag in range(self.num_bags):
            model = model_constructor()
            data_bag, target_bag = data[self.indices_list[bag]], target[self.indices_list[bag]]
            self.models_list.append(model.fit(data_bag, target_bag)) # store fitted models here
        if self.oob:
            self.data = data
            self.target = target
        
    def predict(self, data):
        '''
        Get average prediction for every object from passed dataset
        '''
        results = []
        for x in data:
            predictions = []
            for model in self.models_list:
                prediction = model.predict(np.expand_dims(x, 0))
                mean_prediction = np.mean(prediction)
                predictions.append(mean_prediction)
            mean_predictions = np.mean(predictions)
            results.append(mean_predictions)
        return results
    
    def _get_oob_predictions_from_every_model(self):
        '''
        Generates list of lists, where list i contains predictions for self.data[i] object
        from all models, which have not seen this object during training phase
        '''
        list_of_predictions_lists = [[] for _ in range(len(self.data))]
        self.list_of_predictions_lists = []
        for i in range(len(self.data)):
            predictions_list = []
            for model in np.array(self.models_list)[np.array([i not in item for item in self.indices_list])]:
                prediction = model.predict(self.data[i].reshape(1, -1)).item()
                predictions_list.append(prediction)
                
            self.list_of_predictions_lists.append(predictions_list)
        self.list_of_predictions_lists = np.array(self.list_of_predictions_lists, dtype=object)

    def _get_averaged_oob_predictions(self):
        '''
        Compute average prediction for every object from training set.
        If object has been used in all bags on training phase, return None instead of prediction
        '''
        self._get_oob_predictions_from_every_model()
        self.oob_predictions = [np.mean(el) if len(el) > 0 else None for el in self.list_of_predictions_lists]
        
        
    def OOB_score(self):
        '''
        Compute mean square error for all objects, which have at least one prediction
        '''
        self._get_averaged_oob_predictions()
        oob_errors = []
        for i in range(len(self.data)):
            if len(self.list_of_predictions_lists[i]) > 0:
                oob_errors.append((self.target[i] - self.oob_predictions[i])**2)
        return np.mean(oob_errors)
