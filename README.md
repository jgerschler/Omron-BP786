# Omron-BP786
Scripts for pulling data from the Omron BP786 blood pressure monitor using LCD digit recognition.

Currently a simple proof of concept based on images only. Time permitting, video capability will be added in the future.

Rudimentary use:

bp = OmronBP786()# start class instance
bp.acquire()# acquire data from image

# These functions each return a tuple of the form (value (int), measuring unit (str))
print(bp.systolic())
print(bp.diastolic())
print(bp.pulse())