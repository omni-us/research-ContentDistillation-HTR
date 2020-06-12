import sys
from main_run import test, all_data_loader


epoch = sys.argv[1]
model_file = 'save_weights/contran-'+epoch+'.model'

_, test_loader = all_data_loader()
test(test_loader, int(epoch), model_file)
