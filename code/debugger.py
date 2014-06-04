from cv_server import Cross_Validator, default_preproc_params

cv_server = Cross_Validator(outofcore=False)
print(cv_server.preprocess(default_preproc_params))

# su_mm doesn't work...??? <-- ZeroDivisionError
# also empty slice in merit I think