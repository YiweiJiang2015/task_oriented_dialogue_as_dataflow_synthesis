#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
from pathlib import Path
import sys, re
# add smcalflow library to path
project_path = Path(re.sub(r'/dataflow.*', '', __file__))
smcalflow_lib_path = project_path #/ 'task_oriented_dialogue_as_dataflow_synthesis/src'
sys.path.append(str(smcalflow_lib_path.resolve()))