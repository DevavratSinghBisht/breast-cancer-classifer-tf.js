// this in and IIFE -> Immidiately Invoked Function Expression
// IIFE is not much useful in this model, but IIFEs are used for data privacy


(function (window) {

	var ref = {};

	ref.run = async function (){

		const train_url = 'dataset/bc_train.csv';	//loc of train data in server
		const test_url = 'dataset/bc_test.csv';		//loc of test data in server


		// We want to predict the column "diagnosis"
		//where 1 indicates a malignant cancer, and a 0 indicates a benign one
		const train_data = tf.data.csv(train_url, {
			columnConfigs: {
				diagnosis: {
					isLabel: true
				}
			}
		});


		// Number of features is the number of column names minus one for the label column.
   		const no_of_features = (await train_data.columnNames()).length - 1
        console.log("No of features: " + no_of_features);


        // Prepare the Dataset for training.
		const flattened_train_data = train_data.map(({xs, ys}) => {

			// Convert xs(features) and ys(labels) from object form (keyed bycolumn name) to array form.
			return{xs: Object.values(xs), ys:Object.values(ys)}
		}).batch(10);



		const test_data = tf.data.csv(test_url, {
			columnConfigs: {
				diagnosis: {
					isLabel: true
				}
			}
		});

		const flattened_test_data = test_data.map(({xs, ys}) => {
			return{xs: Object.values(xs), ys:Object.values(ys)}
		}).batch(10);


		// Define the model.
		const model = tf.sequential();

		model.add(tf.layers.dense({units:32, inputShape: [no_of_features], activation: 'relu'}));
		model.add(tf.layers.dense({units: 1, activation: "sigmoid"}));
		console.log(model.summary());

		model.compile({loss:'binaryCrossentropy',
					optimizer: tf.train.rmsprop(0.05),
					metrics: ['accuracy']});



		// Fit the model using the prepared Dataset
		await model.fitDataset(flattened_train_data, 
                             {epochs:100,
                              validationData: flattened_test_data,
                              callbacks:{
                                  onEpochEnd: async(epoch, logs) =>{
                                      console.log("Epoch: " + epoch + " Loss: " + logs.loss + " Accuracy: " + logs.acc);
                                  }
                              }});
        
        await model.save('downloads://my_model');

	}

	window.ref = ref
})(window);