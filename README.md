# Exporting the Mask RCNN

Currently, this repo contains the code necessary to export [Matterport's Mask RCNN][1] into a TF-Serving deployment ready format. I've also added the tools to serve the exported model locally in TF-Serving (REST and GRPC interface).

**The repo will be expanded to include code for deploying and scaling the Mask RCNN  on Google Cloud in the near future. So don't forget to check back in!!!**

This repository includes the code to:
* Freeze the keras/tensorflow Mask RCNN model
* Apply optional graph optimizations such as weights quantization
* Export into the SavedModel format to be used for TF-Serving
* Build a Docker image based on top of TF-Serving
* Jupyter Notebook to test out the model
* Bonus: Basic Flask-based Mask RCNN web app for comparison purposes

# Getting Started

* Make sure that your Python 3 environment has all of the requirements.
* Edit the [export config][2] file. You can add/remove optional graph optimizations.
* Go to the [tf_serving][3] folder and run `python3 make_tf_serving.py`. This will export the Keras/Tensorflow MaskRCNN into the SavedModel format and apply any optional graph optimizations. 
* (Optional): In the same folder, run `build_image.sh` to build and run a docker image serving the exported model. If you don't have a GPU, remove `_gpu` on the first line of the docker file.
* (Optional): Play around with the [Jupyter Notebook][4] to call the served models and visualize the results.

### Optional
If you wish to try the basic Flask API. Run the following commands:
* `export FLASK_APP=app.py`
* `flask run`
The code for testing out the web app can also be found in the [notebook][4].


[1]: https://github.com/matterport/Mask_RCNN "Mask RCNN"
[2]: https://github.com/moganesyan/tensorflow_model_deployment/blob/mask-r-cnn/tf_serving/export_config.py "export_config"
[3]: https://github.com/moganesyan/tensorflow_model_deployment/tree/mask-r-cnn/tf_serving "tf_serving"
[4]: https://github.com/moganesyan/tensorflow_model_deployment/blob/mask-r-cnn/notebooks/TF_SERVING_Model_Test.ipynb "notebook"
