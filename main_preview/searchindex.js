Search.setIndex({"docnames": ["api_ref_decoders", "api_ref_samplers", "api_ref_torchcodec", "generated/torchcodec.Frame", "generated/torchcodec.FrameBatch", "generated/torchcodec.decoders.VideoDecoder", "generated/torchcodec.decoders.VideoStreamMetadata", "generated/torchcodec.samplers.clips_at_random_indices", "generated/torchcodec.samplers.clips_at_random_timestamps", "generated/torchcodec.samplers.clips_at_regular_indices", "generated/torchcodec.samplers.clips_at_regular_timestamps", "generated_examples/basic_example", "generated_examples/index", "generated_examples/sampling", "generated_examples/sg_execution_times", "glossary", "index", "install_instructions", "sg_execution_times"], "filenames": ["api_ref_decoders.rst", "api_ref_samplers.rst", "api_ref_torchcodec.rst", "generated/torchcodec.Frame.rst", "generated/torchcodec.FrameBatch.rst", "generated/torchcodec.decoders.VideoDecoder.rst", "generated/torchcodec.decoders.VideoStreamMetadata.rst", "generated/torchcodec.samplers.clips_at_random_indices.rst", "generated/torchcodec.samplers.clips_at_random_timestamps.rst", "generated/torchcodec.samplers.clips_at_regular_indices.rst", "generated/torchcodec.samplers.clips_at_regular_timestamps.rst", "generated_examples/basic_example.rst", "generated_examples/index.rst", "generated_examples/sampling.rst", "generated_examples/sg_execution_times.rst", "glossary.rst", "index.rst", "install_instructions.rst", "sg_execution_times.rst"], "titles": ["torchcodec.decoders", "torchcodec.samplers", "torchcodec", "Frame", "FrameBatch", "VideoDecoder", "VideoStreamMetadata", "clips_at_random_indices", "clips_at_random_timestamps", "clips_at_regular_indices", "clips_at_regular_timestamps", "Decoding a video with VideoDecoder", "Interactive examples", "How to sample video clips", "Computation times", "Glossary", "Welcome to the TorchCodec documentation!", "Installation Instructions", "Computation times"], "terms": {"For": [0, 1, 7, 9, 10, 13], "tutori": [0, 1, 13], "see": [0, 1, 13], "video": [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18], "videodecod": [0, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 18], "how": [1, 4, 5, 7, 8, 9, 10, 11, 12, 14, 16, 18], "sampl": [1, 4, 5, 7, 8, 9, 10, 12, 14, 16, 18], "clip": [1, 4, 5, 7, 8, 9, 10, 12, 14, 15, 16, 18], "class": [3, 4, 5, 6, 11], "torchcodec": [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17], "data": [3, 4, 5, 7, 8, 9, 10, 11, 13, 16], "tensor": [3, 4, 5, 11, 13, 16], "pts_second": [3, 4, 7, 8, 9, 10, 11, 13], "float": [3, 4, 5, 6, 8, 10, 13], "duration_second": [3, 4, 6, 7, 8, 9, 10, 11, 13], "sourc": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], "A": [3, 5, 13, 15, 16], "singl": [3, 5, 6, 11], "associ": [3, 4], "metadata": [3, 4, 5, 6, 11, 13, 15], "exampl": [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16, 18], "us": [3, 4, 5, 6, 7, 8, 9, 10, 13, 16], "decod": [3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18], "The": [3, 4, 5, 7, 8, 9, 10, 11, 13, 15, 17], "3": [3, 11, 13], "d": [3, 11], "torch": [3, 4, 5, 11, 13], "durat": [3, 4, 6, 8, 10, 13], "second": [3, 4, 5, 6, 8, 10, 11, 13, 15], "pt": [3, 4, 5, 6, 15], "multipl": [4, 5], "frame": [4, 5, 6, 7, 8, 9, 10, 13, 15], "typic": [4, 13], "4d": [4, 15], "sequenc": [4, 13, 15], "nhwc": [4, 5], "nchw": [4, 5], "5d": [4, 7, 8, 9, 10, 13, 15], "return": [4, 5, 7, 8, 9, 10, 11, 13, 15, 16], "sampler": [4, 7, 8, 9, 10, 15], "when": [4, 5, 7, 8, 9, 10, 11, 13], "resp": 4, "ar": [4, 5, 7, 8, 9, 10, 11, 13, 15, 16, 17], "1d": 4, "2d": [4, 13], "uint8": [4, 11, 13], "union": 5, "str": [5, 6, 7, 8, 9, 10, 11, 13], "path": [5, 11, 13], "byte": [5, 11], "stream_index": [5, 6, 11], "option": [5, 6, 7, 8, 9, 10, 11, 13], "int": [5, 6, 7, 8, 9, 10, 11], "none": [5, 6, 7, 8, 9, 10, 11, 13], "dimension_ord": [5, 7, 8, 9, 10, 11, 13], "liter": [5, 7, 8, 9, 10], "num_ffmpeg_thread": 5, "1": [5, 6, 7, 8, 9, 10, 11, 13], "devic": 5, "cpu": 5, "stream": [5, 6, 15], "thi": [5, 6, 7, 8, 9, 10, 11, 13], "alwai": [5, 11, 13], "perform": [5, 6, 13], "scan": [5, 6, 15], "paramet": [5, 7, 8, 9, 10, 11], "pathlib": 5, "If": [5, 6, 7, 8, 9, 10, 13, 16, 17], "local": [5, 11, 13], "file": [5, 11, 13, 14, 15, 18], "object": [5, 6, 11, 13], "raw": [5, 11], "encod": [5, 11], "specifi": 5, "which": [5, 7, 8, 9, 10, 11, 13, 15], "from": [5, 6, 7, 8, 9, 10, 11, 13, 14, 17, 18], "note": [5, 6, 7, 8, 9, 10, 13, 17], "index": [5, 6, 7, 9], "absolut": 5, "across": [5, 6, 13], "all": [5, 6, 12, 13, 17, 18], "media": 5, "type": [5, 7, 8, 9, 10, 11, 13], "left": 5, "unspecifi": 5, "best": [5, 6, 7, 8, 9, 10, 15], "dimens": [5, 11, 13], "order": [5, 11, 15], "can": [5, 11, 13], "either": [5, 13], "default": [5, 7, 8, 9, 10, 11, 13], "where": [5, 7, 8, 9, 10, 11, 13], "n": [5, 11], "batch": [5, 11, 13], "size": [5, 11, 13], "c": [5, 7, 8, 9, 10, 11, 13, 17], "number": [5, 6, 7, 8, 9, 10, 11], "channel": [5, 11], "h": [5, 7, 8, 9, 10, 11, 13], "height": [5, 6, 11], "w": [5, 7, 8, 9, 10, 11, 13], "width": [5, 6, 11], "nativ": [5, 13], "format": 5, "underli": 5, "ffmpeg": [5, 15, 16, 17], "implement": 5, "convert": 5, "those": [5, 7, 8, 9, 10, 13, 16], "cheap": 5, "copi": 5, "oper": 5, "allow": [5, 13], "transform": [5, 11, 13, 16], "torchvis": [5, 10, 11, 13], "http": [5, 11, 13], "pytorch": [5, 13, 16, 17], "org": 5, "vision": 5, "stabl": [5, 17], "html": [5, 16], "_": 5, "thread": 5, "mai": [5, 7, 8, 9, 10, 13, 15, 17], "you": [5, 10, 11, 13, 16, 17], "run": [5, 11, 13], "instanc": [5, 7, 8, 9, 10], "parallel": 5, "higher": 5, "multi": 5, "variabl": [5, 13], "videostreammetadata": [5, 11], "retriev": [5, 6, 15], "wa": [5, 6, 7, 8, 9, 10], "provid": 5, "initi": 5, "same": [5, 13], "valu": [5, 6, 7, 8, 9, 10, 11, 13], "__getitem__": 5, "kei": 5, "integr": [5, 16], "slice": [5, 11], "given": [5, 13], "rang": [5, 7, 8, 9, 10], "s": [5, 6, 13, 16, 17], "get_frame_at": [5, 11], "get_frame_played_at": [5, 6, 11], "plai": [5, 6, 11, 15], "timestamp": [5, 8, 10, 13], "time": [5, 6, 8, 10, 15, 17], "stamp": [5, 11, 15], "get_frames_at": 5, "indic": [5, 7, 9, 13], "list": 5, "framebatch": [5, 7, 8, 9, 10, 11, 13, 15], "call": [5, 6], "method": [5, 11], "more": [5, 6, 8, 10, 11, 13], "effici": [5, 13, 16], "repeat": [5, 7, 8, 9, 10, 13], "individu": 5, "make": [5, 13], "sure": 5, "twice": 5, "also": [5, 11, 13], "avoid": 5, "backward": 5, "seek": [5, 16], "slow": 5, "get_frames_in_rang": [5, 11], "start": [5, 7, 8, 9, 10, 11, 13, 16], "stop": [5, 11], "step": [5, 11, 17], "first": [5, 6, 7, 8, 9, 10, 11, 13], "end": [5, 6, 7, 8, 9, 10, 13], "exclus": [5, 7, 8, 9, 10], "per": [5, 7, 8, 9, 10, 11], "python": [5, 11, 12, 13, 16], "convent": [5, 16], "between": [5, 7, 8, 9, 10], "within": [5, 6, 7, 8, 9, 10, 13], "get_frames_played_at": 5, "get_frames_played_in_rang": [5, 11], "start_second": [5, 11], "stop_second": [5, 11], "half": 5, "open": [5, 10, 13, 16], "each": [5, 8, 10, 13], "insid": 5, "As": 5, "exclud": 5, "duration_seconds_from_head": [6, 11], "bit_rat": [6, 11], "num_frames_from_head": [6, 11], "num_frames_from_cont": [6, 11], "begin_stream_second": [6, 11], "end_stream_second": [6, 11, 13], "codec": [6, 11], "average_fps_from_head": [6, 11], "properti": 6, "average_fp": [6, 8, 10, 11], "averag": [6, 8, 10], "fp": [6, 13], "perfom": 6, "comput": 6, "otherwis": 6, "we": [6, 7, 8, 9, 10, 11, 13, 16, 17], "fall": [6, 13], "back": 6, "aver": 6, "obtain": 6, "header": [6, 11, 13], "recommend": [6, 13], "attribut": [6, 11], "instead": [6, 10], "begin": [6, 7, 8, 9, 10], "conceptu": 6, "correspond": [6, 8, 10, 15], "It": [6, 15, 16], "min": 6, "usual": [6, 15], "equal": [6, 9, 10], "0": [6, 7, 8, 9, 10, 11, 13, 14, 18], "bit": [6, 11, 13], "rate": 6, "try": [6, 11, 13], "calcul": 6, "actual": 6, "could": 6, "inaccur": 6, "last_fram": [6, 11], "max": 6, "so": [6, 7, 8, 9, 10, 11, 13, 15], "would": [6, 7, 8, 9, 10, 13], "result": [6, 8, 10], "an": [6, 7, 8, 9, 10, 13, 15, 16], "error": [6, 7, 8, 9, 10], "last": [6, 7, 8, 9, 10, 11, 13], "done": 6, "simpli": [6, 13], "num_fram": [6, 11], "made": 6, "content": [6, 11, 13], "doesn": [6, 13], "t": [6, 13], "involv": [6, 15], "accur": [6, 8, 10], "than": [6, 11, 15], "potenti": 6, "num_clip": [7, 8, 9, 10, 13], "num_frames_per_clip": [7, 8, 9, 10, 13], "num_indices_between_fram": [7, 9, 13], "sampling_range_start": [7, 8, 9, 10, 13], "sampling_range_end": [7, 8, 9, 10, 13], "polici": [7, 8, 9, 10], "repeat_last": [7, 8, 9, 10, 13], "wrap": [7, 8, 9, 10, 13], "random": [7, 8, 13], "mean": [7, 9, 13], "consecut": [7, 9, 15], "sometim": [7, 9, 11, 13], "refer": [7, 9], "dilat": [7, 9], "defin": [7, 8, 9, 10, 11, 13], "i": [7, 8, 9, 10], "e": [7, 8, 9, 10], "onli": [7, 8, 9, 10, 11, 13, 17], "set": [7, 8, 9, 10, 11, 13], "automat": [7, 8, 9, 10, 13], "never": [7, 8, 9, 10], "span": [7, 8, 9, 10], "beyond": [7, 8, 9, 10, 13], "valid": [7, 8, 9, 10, 13], "99": [7, 9], "10": [7, 8, 9, 10, 11, 13], "90": [7, 9], "neg": [7, 8, 9, 10], "accept": [7, 9], "equival": [7, 9], "len": [7, 9, 11], "val": [7, 9], "construct": [7, 8, 9, 10, 13], "describ": [7, 8, 9, 10], "assum": [7, 8, 9, 10], "95": [7, 9], "5": [7, 8, 9, 10, 11, 13], "2": [7, 8, 9, 10, 11, 13, 14, 17, 18], "suppos": [7, 8, 9, 10], "97": [7, 9], "101": [7, 9], "103": [7, 9], "But": [7, 8, 9, 10], "invalid": [7, 8, 9, 10, 13], "replac": [7, 8, 9, 10, 13], "get": [7, 8, 9, 10, 11, 13, 16], "around": [7, 8, 9, 10, 13], "rais": [7, 8, 9, 10, 11, 13], "unlik": [7, 8, 9, 10], "relev": [7, 8, 9, 10], "shape": [7, 8, 9, 10, 11, 13], "field": [7, 8, 9, 10, 11, 13], "depend": [7, 8, 9, 10, 13], "seconds_between_fram": [8, 10, 13], "point": [8, 10], "becaus": [8, 10, 13], "interv": [8, 10, 13], "exactli": [8, 10, 13], "space": [8, 9, 10], "thei": [8, 10, 11, 13], "some": [8, 10, 11, 13], "why": [8, 10], "end_video_second": [8, 10], "seekabl": [8, 10], "9": [8, 10, 13], "7": [8, 10, 17], "11": [8, 10, 13], "regular": [9, 10], "seconds_between_clip_start": [10, 13], "consist": 10, "exist": 10, "api": [10, 13, 16], "take": [10, 13], "find": 10, "support": [10, 13, 17], "pleas": [10, 11, 13, 16], "let": [10, 13, 16], "know": [10, 16], "featur": 10, "request": [10, 11, 13], "In": [11, 13, 15], "ll": [11, 13], "learn": [11, 13], "boilerpl": [11, 13], "download": [11, 12, 13], "web": [11, 13], "plot": [11, 13], "util": [11, 13], "ignor": [11, 13], "part": [11, 13], "jump": [11, 13], "right": [11, 13], "below": [11, 13], "import": [11, 13], "www": [11, 13], "pexel": [11, 13], "com": [11, 13], "dog": [11, 13], "eat": [11, 13], "854132": [11, 13], "licens": [11, 13], "cc0": [11, 13], "author": [11, 13], "coverr": [11, 13], "url": [11, 13], "sd_640_360_25fp": [11, 13], "mp4": [11, 13], "respons": [11, 13], "user": [11, 13, 15], "agent": [11, 13], "status_cod": [11, 13], "200": [11, 13], "runtimeerror": [11, 13], "f": [11, 13], "fail": [11, 13], "raw_video_byt": [11, 13], "def": [11, 13], "titl": [11, 13], "make_grid": [11, 13], "v2": [11, 13], "function": [11, 13], "to_pil_imag": [11, 13], "matplotlib": [11, 13], "pyplot": [11, 13], "plt": [11, 13], "except": [11, 13], "importerror": [11, 13], "print": [11, 13], "cannot": [11, 13], "pip": [11, 13, 17], "instal": [11, 13, 16], "rcparam": [11, 13], "savefig": [11, 13], "bbox": [11, 13], "tight": [11, 13], "fig": [11, 13], "ax": [11, 13], "subplot": [11, 13], "imshow": [11, 13], "xticklabel": [11, 13], "yticklabel": [11, 13], "xtick": [11, 13], "ytick": [11, 13], "set_titl": [11, 13], "tight_layout": [11, 13], "now": [11, 13, 17], "cours": 11, "pass": [11, 13, 15], "input": [11, 13], "rather": 11, "ha": 11, "yet": 11, "been": 11, "alreadi": [11, 13, 16, 17], "have": [11, 13, 16], "access": 11, "via": 11, "345": 11, "13": [11, 13], "8": [11, 13], "25": 11, "505790": 11, "h264": 11, "640": [11, 13], "360": [11, 13], "first_fram": 11, "every_twenty_fram": 11, "20": 11, "dtype": [11, 13], "18": 11, "By": [11, 13], "present": [11, 15], "re": [11, 13], "one": 11, "chang": [11, 17], "everi": 11, "normal": 11, "like": [11, 15], "assert": 11, "isinst": 11, "pure": 11, "addit": [11, 13], "inform": 11, "about": [11, 13, 15], "achiev": [11, 16], "respect": 11, "_frame": 11, "76": 11, "04": 11, "middle_fram": 11, "4000": 11, "4800": [11, 13], "5600": [11, 13], "6400": 11, "7200": [11, 13], "float64": [11, 13], "0400": [11, 13], "middl": 11, "both": [11, 13], "contain": [11, 13], "far": [11, 13], "frame_at_2_second": 11, "first_two_second": 11, "50": 11, "0000": [11, 13], "0800": 11, "1200": 11, "1600": [11, 13], "2000": [11, 13], "2400": 11, "2800": [11, 13], "3200": [11, 13], "3600": [11, 13], "4400": [11, 13], "5200": 11, "6000": [11, 13], "6800": [11, 13], "7600": 11, "8000": [11, 13], "8400": [11, 13], "8800": 11, "9200": [11, 13], "9600": [11, 13], "dure": 11, "total": [11, 13, 14, 18], "script": [11, 13], "minut": [11, 13], "113": [11, 14, 18], "jupyt": [11, 12, 13], "notebook": [11, 12, 13], "basic_exampl": [11, 14, 16, 18], "ipynb": [11, 13], "code": [11, 12, 13], "py": [11, 13, 14, 18], "zip": [11, 12, 13], "galleri": [11, 12, 13, 18], "gener": [11, 12, 13], "sphinx": [11, 12, 13], "generated_examples_python": 12, "generated_examples_jupyt": 12, "denot": 13, "model": [13, 16], "familiar": 13, "quick": 13, "look": 13, "our": [13, 16], "simpl": [13, 16], "other": [13, 17], "follow": [13, 17], "similar": 13, "principl": 13, "clips_at_random_indic": 13, "rng": 13, "control": 13, "seed": 13, "reproduc": 13, "hard": 13, "train": [13, 16], "manual_se": 13, "4": [13, 17], "output": 13, "repres": [13, 15], "differ": [13, 15], "Its": 13, "what": [13, 15], "give": 13, "semant": 13, "includ": 13, "fanci": 13, "easi": [13, 16], "filter": 13, "criteria": 13, "abov": 13, "easili": 13, "out": 13, "who": 13, "after": 13, "specif": 13, "clip_start": 13, "clips_starting_after_five_second": 13, "every_other_clip": 13, "natur": 13, "wai": 13, "reli": [13, 16], "cover": 13, "later": [13, 17], "ve": 13, "under": 13, "two": 13, "main": 13, "categori": 13, "clips_at_regular_indic": 13, "clips_at_random_timestamp": 13, "clips_at_regular_timestamp": 13, "analog": 13, "ones": 13, "offer": 13, "compar": 13, "term": 13, "speed": 13, "Is": 13, "better": 13, "arguabl": 13, "slightli": [13, 17], "simpler": 13, "behavior": 13, "possibl": 13, "understand": 13, "discret": 13, "constant": 13, "behav": 13, "howev": [13, 16], "often": 13, "case": 13, "over": [13, 15, 17], "region": 13, "lead": 13, "undersir": 13, "side": 13, "effect": 13, "ensur": [13, 16], "uniform": 13, "caracterist": 13, "along": 13, "want": [13, 16], "entir": [13, 15], "interest": 13, "smaller": 13, "parmet": 13, "There": [13, 17], "thing": 13, "keep": 13, "mind": 13, "upper": 13, "bound": 13, "length": 13, "should": [13, 15, 17], "end_of_video": 13, "8s": 13, "tri": 13, "28": 13, "68": 13, "14": 13, "08": 13, "With": 13, "altern": 13, "few": 13, "necessari": 13, "earli": [13, 16], "enough": 13, "befor": 13, "rare": 13, "come": [13, 17], "action": 13, "most": [13, 15, 16, 17], "probabl": [13, 17], "don": 13, "need": [13, 17], "worri": 13, "too": 13, "much": 13, "399": [13, 14, 18], "00": [14, 18], "03": [14, 18], "512": [14, 18], "execut": [14, 18], "generated_exampl": [14, 16], "mem": [14, 18], "mb": [14, 18], "express": 15, "notion": 15, "determin": 15, "quot": 15, "doc": 15, "accord": 15, "variou": 15, "heurist": 15, "expect": 15, "purpos": 15, "doe": 15, "lot": 15, "cheaper": 15, "necessarili": 15, "group": 15, "librari": 16, "aim": 16, "fast": 16, "well": 16, "ecosystem": 16, "ml": 16, "turn": 16, "capabl": 16, "through": 16, "mirror": 16, "do": 16, "version": [16, 17], "fmpeg": 16, "matur": 16, "broad": 16, "coverag": 16, "avail": [16, 17], "system": 16, "abstract": 16, "complex": 16, "correctli": 16, "readi": 16, "fed": 16, "directli": 16, "still": 16, "develop": 16, "stage": 16, "activ": 16, "feedback": 16, "ani": 16, "suggest": 16, "issu": 16, "github": 16, "repositori": 16, "instruct": 16, "install_instruct": 16, "linux": 17, "plan": 17, "platform": 17, "futur": 17, "three": 17, "latest": 17, "offici": 17, "requir": 17, "your": 17, "distribut": 17, "pre": 17, "major": 17, "conda": 17, "forg": 17, "up": 17, "date": 17, "readm": 17}, "objects": {"torchcodec": [[3, 0, 1, "", "Frame"], [4, 0, 1, "", "FrameBatch"]], "torchcodec.Frame": [[3, 1, 1, "", "data"], [3, 1, 1, "", "duration_seconds"], [3, 1, 1, "", "pts_seconds"]], "torchcodec.FrameBatch": [[4, 1, 1, "", "data"], [4, 1, 1, "", "duration_seconds"], [4, 1, 1, "", "pts_seconds"]], "torchcodec.decoders": [[5, 0, 1, "", "VideoDecoder"], [6, 0, 1, "", "VideoStreamMetadata"]], "torchcodec.decoders.VideoDecoder": [[5, 2, 1, "", "__getitem__"], [5, 2, 1, "", "get_frame_at"], [5, 2, 1, "", "get_frame_played_at"], [5, 2, 1, "", "get_frames_at"], [5, 2, 1, "", "get_frames_in_range"], [5, 2, 1, "", "get_frames_played_at"], [5, 2, 1, "", "get_frames_played_in_range"]], "torchcodec.decoders.VideoStreamMetadata": [[6, 3, 1, "", "average_fps"], [6, 1, 1, "", "average_fps_from_header"], [6, 1, 1, "", "begin_stream_seconds"], [6, 1, 1, "", "bit_rate"], [6, 1, 1, "", "codec"], [6, 3, 1, "", "duration_seconds"], [6, 1, 1, "", "duration_seconds_from_header"], [6, 1, 1, "", "end_stream_seconds"], [6, 1, 1, "", "height"], [6, 3, 1, "", "num_frames"], [6, 1, 1, "", "num_frames_from_content"], [6, 1, 1, "", "num_frames_from_header"], [6, 1, 1, "", "stream_index"], [6, 1, 1, "", "width"]], "torchcodec.samplers": [[7, 4, 1, "", "clips_at_random_indices"], [8, 4, 1, "", "clips_at_random_timestamps"], [9, 4, 1, "", "clips_at_regular_indices"], [10, 4, 1, "", "clips_at_regular_timestamps"]]}, "objtypes": {"0": "py:class", "1": "py:attribute", "2": "py:method", "3": "py:property", "4": "py:function"}, "objnames": {"0": ["py", "class", "Python class"], "1": ["py", "attribute", "Python attribute"], "2": ["py", "method", "Python method"], "3": ["py", "property", "Python property"], "4": ["py", "function", "Python function"]}, "titleterms": {"torchcodec": [0, 1, 2, 16], "decod": [0, 11, 13], "sampler": [1, 13], "frame": [3, 11], "framebatch": 4, "videodecod": [5, 11], "videostreammetadata": 6, "clips_at_random_indic": 7, "clips_at_random_timestamp": 8, "clips_at_regular_indic": 9, "clips_at_regular_timestamp": 10, "video": [11, 13], "creat": [11, 13], "index": [11, 13], "iter": 11, "over": 11, "retriev": 11, "pt": 11, "durat": 11, "us": 11, "time": [11, 13, 14, 18], "base": [11, 13], "interact": 12, "exampl": 12, "how": 13, "sampl": 13, "clip": 13, "basic": 13, "manipul": 13, "advanc": 13, "paramet": 13, "rang": 13, "polici": 13, "comput": [14, 18], "glossari": 15, "welcom": 16, "document": 16, "instal": 17, "instruct": 17}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinx.ext.todo": 2, "sphinx.ext.viewcode": 1, "sphinx": 56}})