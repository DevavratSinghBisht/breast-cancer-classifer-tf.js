# my.javascript-tesnsorflow

## 1) Breast Cancer Classifier

	Code is located at script/bc_class.js
	Dataset is located at dataset directory.

	Once the program runs:
		1)The model is automatically downloaded as my_model.json
		2)Weights of model is automatically downloaded as my_model.weights.bin.
		3)Click on allow, if browsers shows an alert about multiple downloads.

	Outputs:
		1)No of features in dataset.
		2)Summary of the model.
		3)Loss and Accuray for each epoch,
	can be seen in console.

	Information about implemented Model:
		1) 2 layer architecture 1st has relu activation with 32 units and 2nd has a single sigmoid unit.
		2)Loss: Binary Crossentropy
		3)Optimizer: RMSProp
		4)Learning Rate: 0.05
		5)Metrics: Accuracy
		6)Epochs: 100
		7)Callpacks: On Epoch End, used to print loss and accuracy in console.
		
## 2) Fashion MNIST Classifier
	
	Runs fine for some time and then the WebGl context is lost.
