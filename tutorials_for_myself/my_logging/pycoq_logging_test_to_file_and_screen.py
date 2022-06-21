# import pycoq.log
#
# def create_config():
#     pycoq_config = defaultdict(None, {
#         "opam_root": None,
#         "log_level": 4,
#         "log_filename": os.path.join(os.getenv('HOME'), 'pycoq.log')
#     })
#     # create a clean version of the log file
#     print(f'--> Path to our pycoq config file: {pycoq.config.PYCOQ_CONFIG_FILE=}')
#     with open(pycoq.config.PYCOQ_CONFIG_FILE, 'w+') as f:
#         json.dump(pycoq_config, f, indent=4, sort_keys=True)
#     print('Print contents of our pycoq config file:')
#     pprint(pycoq_config)
