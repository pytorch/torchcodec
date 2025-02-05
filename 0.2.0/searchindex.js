Search.setIndex({"docnames": ["api_ref_decoders", "api_ref_samplers", "api_ref_torchcodec", "generated/torchcodec.Frame", "generated/torchcodec.FrameBatch", "generated/torchcodec.decoders.VideoDecoder", "generated/torchcodec.decoders.VideoStreamMetadata", "generated/torchcodec.samplers.clips_at_random_indices", "generated/torchcodec.samplers.clips_at_random_timestamps", "generated/torchcodec.samplers.clips_at_regular_indices", "generated/torchcodec.samplers.clips_at_regular_timestamps", "generated_examples/approximate_mode", "generated_examples/basic_cuda_example", "generated_examples/basic_example", "generated_examples/index", "generated_examples/sampling", "generated_examples/sg_execution_times", "glossary", "index", "sg_execution_times"], "filenames": ["api_ref_decoders.rst", "api_ref_samplers.rst", "api_ref_torchcodec.rst", "generated/torchcodec.Frame.rst", "generated/torchcodec.FrameBatch.rst", "generated/torchcodec.decoders.VideoDecoder.rst", "generated/torchcodec.decoders.VideoStreamMetadata.rst", "generated/torchcodec.samplers.clips_at_random_indices.rst", "generated/torchcodec.samplers.clips_at_random_timestamps.rst", "generated/torchcodec.samplers.clips_at_regular_indices.rst", "generated/torchcodec.samplers.clips_at_regular_timestamps.rst", "generated_examples/approximate_mode.rst", "generated_examples/basic_cuda_example.rst", "generated_examples/basic_example.rst", "generated_examples/index.rst", "generated_examples/sampling.rst", "generated_examples/sg_execution_times.rst", "glossary.rst", "index.rst", "sg_execution_times.rst"], "titles": ["torchcodec.decoders", "torchcodec.samplers", "torchcodec", "Frame", "FrameBatch", "VideoDecoder", "VideoStreamMetadata", "clips_at_random_indices", "clips_at_random_timestamps", "clips_at_regular_indices", "clips_at_regular_timestamps", "Exact vs Approximate seek mode: Performance and accuracy comparison", "Accelerated video decoding on GPUs with CUDA and NVDEC", "Decoding a video with VideoDecoder", "Interactive examples", "How to sample video clips", "Computation times", "Glossary", "Welcome to the TorchCodec documentation!", "Computation times"], "terms": {"For": [0, 1, 7, 9, 10, 11, 15], "tutori": [0, 1, 12, 15], "see": [0, 1, 12, 15], "video": [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 16, 17, 18, 19], "videodecod": [0, 3, 4, 6, 7, 8, 9, 10, 14, 15, 16, 17, 19], "how": [1, 4, 5, 7, 8, 9, 10, 13, 14, 16, 18, 19], "sampl": [1, 4, 5, 7, 8, 9, 10, 14, 16, 18, 19], "clip": [1, 4, 5, 7, 8, 9, 10, 14, 16, 17, 18, 19], "class": [3, 4, 5, 6, 11, 12, 13], "torchcodec": [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17], "data": [3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15, 18], "tensor": [3, 4, 5, 11, 12, 13, 15, 18], "pts_second": [3, 4, 7, 8, 9, 10, 13, 15], "float": [3, 4, 5, 6, 8, 10, 12, 15], "duration_second": [3, 4, 6, 7, 8, 9, 10, 11, 13, 15], "sourc": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], "A": [3, 5, 15, 17, 18], "singl": [3, 5, 6, 13], "associ": [3, 4], "metadata": [3, 4, 5, 6, 13, 15, 17], "exampl": [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 16, 18, 19], "us": [3, 4, 5, 6, 7, 8, 9, 10, 15, 17, 18], "decod": [3, 4, 5, 6, 7, 8, 9, 10, 14, 16, 17, 18, 19], "The": [3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15, 17], "3": [3, 12, 13, 15], "d": [3, 13], "torch": [3, 4, 5, 11, 12, 13, 15], "durat": [3, 4, 6, 8, 10, 11, 15], "second": [3, 4, 5, 6, 8, 10, 11, 12, 13, 15, 17], "pt": [3, 4, 5, 6, 17], "multipl": [4, 5], "frame": [4, 5, 6, 7, 8, 9, 10, 15, 17], "typic": [4, 11, 15], "4d": [4, 17], "sequenc": [4, 15, 17], "nhwc": [4, 5], "nchw": [4, 5, 12], "5d": [4, 7, 8, 9, 10, 15, 17], "return": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 18], "sampler": [4, 7, 8, 9, 10, 11, 17], "when": [4, 5, 7, 8, 9, 10, 11, 13, 15, 17], "resp": 4, "ar": [4, 5, 7, 8, 9, 10, 12, 13, 15, 17, 18], "1d": 4, "2d": [4, 15], "alwai": [4, 5, 11, 13, 15], "cpu": [4, 5, 12, 18], "even": 4, "gpu": [4, 5, 14, 16, 18, 19], "uint8": [4, 12, 13, 15], "union": 5, "str": [5, 6, 7, 8, 9, 10, 13, 15], "path": [5, 11, 13, 15], "byte": [5, 13], "stream_index": [5, 6, 11, 13], "option": [5, 6, 7, 8, 9, 10, 13, 15], "int": [5, 6, 7, 8, 9, 10, 13], "none": [5, 6, 7, 8, 9, 10, 11, 13, 15], "dimension_ord": [5, 7, 8, 9, 10, 13, 15], "liter": [5, 7, 8, 9, 10], "num_ffmpeg_thread": 5, "1": [5, 6, 7, 8, 9, 10, 12, 13, 15], "devic": [5, 12], "seek_mod": [5, 11, 17], "exact": [5, 7, 12, 14, 16, 17, 19], "approxim": [5, 7, 14, 16, 17, 19], "stream": [5, 6, 12, 17], "paramet": [5, 7, 8, 9, 10, 11, 12, 13], "pathlib": [5, 11], "If": [5, 6, 7, 8, 9, 10, 11, 15, 18], "local": [5, 13, 15], "file": [5, 11, 13, 15, 16, 17, 18, 19], "object": [5, 6, 11, 12, 13, 15], "raw": [5, 13], "encod": [5, 12, 13], "specifi": 5, "which": [5, 7, 8, 9, 10, 12, 13, 15, 17], "from": [5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 18, 19], "note": [5, 6, 7, 8, 9, 10, 15], "thi": [5, 6, 7, 8, 9, 10, 12, 13, 15], "index": [5, 6, 7, 9, 11], "absolut": 5, "across": [5, 6, 15], "all": [5, 6, 14, 15, 19], "media": 5, "type": [5, 7, 8, 9, 10, 13, 15], "left": [5, 12], "unspecifi": 5, "best": [5, 6, 7, 8, 9, 10, 12, 17], "dimens": [5, 13, 15], "order": [5, 13, 17], "can": [5, 11, 12, 13, 15], "either": [5, 15], "default": [5, 7, 8, 9, 10, 13, 15], "where": [5, 7, 8, 9, 10, 12, 13, 15], "n": [5, 13], "batch": [5, 12, 13, 15], "size": [5, 12, 13, 15], "c": [5, 7, 8, 9, 10, 11, 13, 15], "number": [5, 6, 7, 8, 9, 10, 13], "channel": [5, 13], "h": [5, 7, 8, 9, 10, 12, 13, 15], "height": [5, 6, 11, 13], "w": [5, 7, 8, 9, 10, 13, 15], "width": [5, 6, 11, 13], "nativ": [5, 15], "format": [5, 12], "underli": 5, "ffmpeg": [5, 11, 12, 17, 18], "implement": 5, "convert": [5, 12], "those": [5, 7, 8, 9, 10, 15, 18], "cheap": 5, "copi": [5, 11], "oper": 5, "allow": [5, 15], "transform": [5, 12, 13, 15, 18], "torchvis": [5, 10, 12, 13, 15], "http": [5, 11, 12, 13, 15, 18], "pytorch": [5, 15, 18], "org": [5, 12], "vision": 5, "stabl": 5, "html": [5, 18], "_": [5, 11], "thread": 5, "mai": [5, 7, 8, 9, 10, 11, 12, 15, 17], "you": [5, 10, 11, 12, 13, 15, 18], "run": [5, 11, 12, 13, 15], "instanc": [5, 7, 8, 9, 10], "parallel": 5, "higher": [5, 11], "multi": 5, "pass": [5, 12, 13, 15, 17], "0": [5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 19], "let": [5, 10, 12, 15, 18], "decid": 5, "determin": [5, 17], "access": [5, 13], "guarante": 5, "request": [5, 10, 11, 12, 13, 15], "i": [5, 7, 8, 9, 10, 12], "do": [5, 12, 18], "so": [5, 6, 7, 8, 9, 10, 12, 13, 15, 17], "requir": [5, 12], "an": [5, 6, 7, 8, 9, 10, 11, 15, 17, 18], "initi": 5, "scan": [5, 6, 11, 17], "faster": [5, 11, 12], "avoid": 5, "less": [5, 12], "accur": [5, 6, 8, 10, 11], "s": [5, 6, 11, 12, 15, 18], "calcul": [5, 6], "probabl": [5, 15], "read": 5, "more": [5, 6, 8, 10, 11, 13, 15], "about": [5, 11, 13, 15, 17], "vs": [5, 7, 14, 16, 19], "seek": [5, 7, 14, 16, 18, 19], "mode": [5, 7, 14, 16, 19], "perform": [5, 6, 7, 14, 15, 16, 17, 19], "accuraci": [5, 7, 14, 16, 19], "comparison": [5, 7, 14, 16, 19], "variabl": [5, 11, 15], "videostreammetadata": [5, 11, 13], "retriev": [5, 6, 17], "wa": [5, 6, 7, 8, 9, 10], "provid": 5, "same": [5, 11, 15], "valu": [5, 6, 7, 8, 9, 10, 13, 15], "acceler": [5, 14, 16, 19], "cuda": [5, 14, 16, 18, 19], "nvdec": [5, 14, 16, 19], "__getitem__": 5, "kei": [5, 11], "integr": [5, 18], "slice": [5, 13], "given": [5, 15], "rang": [5, 7, 8, 9, 10, 11, 12], "get_frame_at": [5, 11, 13], "get_frame_played_at": [5, 6, 13], "plai": [5, 6, 13, 17], "timestamp": [5, 8, 10, 12, 15], "time": [5, 6, 8, 10, 11, 12, 17], "stamp": [5, 13, 17], "get_frames_at": [5, 13], "indic": [5, 7, 9, 15], "list": 5, "framebatch": [5, 7, 8, 9, 10, 13, 15, 17], "call": [5, 6, 12], "method": [5, 13], "effici": [5, 15, 18], "repeat": [5, 7, 8, 9, 10, 11, 15], "individu": 5, "make": [5, 12, 15], "sure": 5, "twice": 5, "also": [5, 11, 12, 13, 15], "backward": 5, "slow": 5, "get_frames_in_rang": 5, "start": [5, 7, 8, 9, 10, 11, 15, 18], "stop": 5, "step": [5, 12], "first": [5, 6, 7, 8, 9, 10, 11, 13, 15], "end": [5, 6, 7, 8, 9, 10, 11, 15], "exclus": [5, 7, 8, 9, 10], "per": [5, 7, 8, 9, 10, 11, 13], "python": [5, 11, 12, 13, 14, 15, 18], "convent": [5, 18], "between": [5, 7, 8, 9, 10, 11], "within": [5, 6, 7, 8, 9, 10, 15], "get_frames_played_at": [5, 12, 13], "get_frames_played_in_rang": 5, "start_second": 5, "stop_second": 5, "half": 5, "open": [5, 10, 11, 15, 18], "each": [5, 8, 10, 15], "insid": 5, "As": 5, "exclud": 5, "duration_seconds_from_head": [6, 11, 13], "bit_rat": [6, 11, 13], "num_frames_from_head": [6, 11, 13], "num_frames_from_cont": [6, 11, 13], "begin_stream_seconds_from_cont": [6, 11, 13], "end_stream_seconds_from_cont": [6, 11, 13], "codec": [6, 11, 12, 13], "average_fps_from_head": [6, 11, 13], "properti": [6, 12], "average_fp": [6, 8, 10, 11, 13], "averag": [6, 8, 10], "fp": [6, 12, 15], "perfom": 6, "comput": 6, "otherwis": 6, "we": [6, 7, 8, 9, 10, 11, 12, 13, 15, 18], "fall": [6, 15], "back": 6, "aver": 6, "obtain": 6, "header": [6, 11, 13, 15], "recommend": [6, 15], "attribut": [6, 13], "instead": [6, 10], "begin_stream_second": 6, "begin": [6, 7, 8, 9, 10], "conceptu": 6, "correspond": [6, 8, 10, 17], "It": [6, 11, 12, 17, 18], "min": [6, 11], "usual": [6, 11, 17], "equal": [6, 9, 10, 12], "bit": [6, 11, 12, 13, 15], "rate": 6, "try": [6, 12, 13, 15], "actual": [6, 12], "could": 6, "inaccur": 6, "end_stream_second": [6, 15], "last_fram": [6, 13], "max": [6, 12], "would": [6, 7, 8, 9, 10, 15], "result": [6, 8, 10, 12], "error": [6, 7, 8, 9, 10], "last": [6, 7, 8, 9, 10, 13, 15], "done": 6, "simpli": [6, 12, 15], "num_fram": [6, 11, 13], "made": 6, "content": [6, 13, 15], "doesn": [6, 11, 12, 15, 17], "t": [6, 11, 12, 15, 17], "involv": [6, 11, 17], "than": [6, 11, 12, 13, 17], "potenti": [6, 11], "num_clip": [7, 8, 9, 10, 11, 15], "num_frames_per_clip": [7, 8, 9, 10, 11, 15], "num_indices_between_fram": [7, 9, 15], "sampling_range_start": [7, 8, 9, 10, 15], "sampling_range_end": [7, 8, 9, 10, 15], "polici": [7, 8, 9, 10], "repeat_last": [7, 8, 9, 10, 15], "wrap": [7, 8, 9, 10, 15], "random": [7, 8, 15, 18], "mean": [7, 9, 11, 12, 15], "consecut": [7, 9, 17], "sometim": [7, 9, 13, 15], "refer": [7, 9, 12], "dilat": [7, 9], "defin": [7, 8, 9, 10, 13, 15], "e": [7, 8, 9, 10, 11, 12], "onli": [7, 8, 9, 10, 11, 13, 15], "set": [7, 8, 9, 10, 13, 15], "automat": [7, 8, 9, 10, 15], "never": [7, 8, 9, 10], "span": [7, 8, 9, 10], "beyond": [7, 8, 9, 10, 15], "valid": [7, 8, 9, 10, 15], "99": [7, 9, 11], "10": [7, 8, 9, 10, 11, 13, 15], "90": [7, 9], "neg": [7, 8, 9, 10], "accept": [7, 9], "equival": [7, 9, 12], "len": [7, 9, 11, 12, 13], "val": [7, 9], "construct": [7, 8, 9, 10, 15], "describ": [7, 8, 9, 10, 11], "assum": [7, 8, 9, 10], "95": [7, 9, 11], "5": [7, 8, 9, 10, 11, 12, 13, 15], "2": [7, 8, 9, 10, 11, 12, 13, 15], "suppos": [7, 8, 9, 10], "97": [7, 9, 12], "101": [7, 9], "103": [7, 9], "But": [7, 8, 9, 10], "invalid": [7, 8, 9, 10, 15], "replac": [7, 8, 9, 10, 15], "get": [7, 8, 9, 10, 11, 13, 15, 18], "around": [7, 8, 9, 10, 15], "rais": [7, 8, 9, 10, 11, 13, 15], "unlik": [7, 8, 9, 10], "relev": [7, 8, 9, 10], "shape": [7, 8, 9, 10, 12, 13, 15], "field": [7, 8, 9, 10, 13, 15], "depend": [7, 8, 9, 10, 15], "seconds_between_fram": [8, 10, 15], "point": [8, 10], "becaus": [8, 10, 11, 12, 15], "interv": [8, 10, 15], "exactli": [8, 10, 15], "space": [8, 9, 10], "thei": [8, 10, 12, 13, 15], "some": [8, 10, 13, 15], "why": [8, 10], "end_video_second": [8, 10], "seekabl": [8, 10], "9": [8, 10, 15], "7": [8, 10, 11, 12], "11": [8, 10, 15], "regular": [9, 10, 18], "seconds_between_clip_start": [10, 15], "consist": 10, "exist": 10, "api": [10, 12, 15, 18], "take": [10, 15], "find": [10, 11], "support": [10, 12, 15], "pleas": [10, 12, 13, 15, 18], "know": [10, 18], "featur": 10, "In": [11, 13, 15, 17], "offer": [11, 12, 15], "trade": 11, "off": 11, "speed": [11, 12, 15], "against": [11, 12], "retreiv": 11, "th": 11, "necessarili": [11, 17], "boilerpl": [11, 13, 15], "ll": [11, 13, 15], "download": [11, 13, 14, 15], "short": 11, "web": [11, 13, 15], "cli": 11, "100": 11, "up": [11, 12], "two": [11, 15], "13": [11, 13, 15], "long": 11, "one": [11, 13], "20": [11, 13], "ignor": [11, 13, 15], "part": [11, 13, 15], "jump": [11, 13, 15], "right": [11, 13, 15], "below": [11, 13, 15], "import": [11, 12, 13, 15], "tempfil": 11, "shutil": 11, "subprocess": 11, "perf_counter_n": 11, "www": [11, 13, 15], "pexel": [11, 13, 15], "com": [11, 13, 15, 18], "dog": [11, 13, 15], "eat": [11, 13, 15], "854132": [11, 13, 15], "licens": [11, 13, 15], "cc0": [11, 13, 15], "author": [11, 13, 15], "coverr": [11, 13, 15], "url": [11, 13, 15], "sd_640_360_25fp": [11, 13, 15], "mp4": [11, 12, 13, 15], "respons": [11, 13, 15], "user": [11, 13, 15, 17], "agent": [11, 13, 15], "status_cod": [11, 13, 15], "200": [11, 13, 15], "runtimeerror": [11, 13, 15], "f": [11, 12, 13, 15], "fail": [11, 13, 15], "temp_dir": 11, "mkdtemp": 11, "short_video_path": 11, "short_video": 11, "wb": 11, "chunk": 11, "iter_cont": 11, "write": 11, "long_video_path": 11, "long_video": 11, "ffmpeg_command": 11, "stream_loop": 11, "check": 11, "true": [11, 12], "stdout": 11, "pipe": 11, "stderr": 11, "print": [11, 12, 13, 15], "60": 11, "minut": [11, 12, 13, 15], "8": [11, 12, 13, 15], "23": 11, "term": [11, 15], "ultim": 11, "affect": 11, "longer": 11, "gain": 11, "def": [11, 12, 13, 15], "bench": 11, "average_ov": 11, "50": [11, 13], "warmup": 11, "f_kwarg": 11, "append": 11, "1e": 11, "6": [11, 12], "ns": 11, "ms": 11, "std": 11, "item": 11, "med": 11, "median": 11, "2f": 11, "creat": 11, "04m": 11, "03": [11, 16, 19], "09m": 11, "114": 11, "68m": 11, "73": 11, "52m": 11, "strictli": 11, "speak": 11, "doe": [11, 17], "have": [11, 12, 13, 15, 18], "direct": 11, "effect": [11, 15], "howev": [11, 15, 18], "pattern": 11, "veri": 11, "well": [11, 18], "sample_clip": 11, "clips_at_random_indic": [11, 15], "299": 11, "06m": 11, "32": 11, "15": 11, "183": 11, "01m": 11, "44": 11, "ve": [11, 15], "seen": 11, "significantli": 11, "price": 11, "pai": 11, "won": 11, "lot": [11, 17], "case": [11, 12, 15], "differ": [11, 12, 15, 17], "net": 11, "win": 11, "exact_decod": 11, "approx_decod": 11, "test": 11, "assert_clos": 11, "atol": 11, "rtol": 11, "345": [11, 13], "25": [11, 13], "505790": [11, 13], "h264": [11, 13], "640": [11, 13, 15], "360": [11, 13, 15], "With": [11, 12, 15], "instanti": 11, "process": 11, "entir": [11, 15, 17], "infer": 11, "like": [11, 12, 13, 17], "build": 11, "intern": 11, "lead": [11, 15], "behavior": [11, 15], "without": 11, "reli": [11, 15, 18], "contain": [11, 13, 15], "gener": [11, 12, 13, 14, 15], "rule": 11, "thumb": 11, "follow": [11, 12, 15], "realli": 11, "care": 11, "sacrific": 11, "your": [11, 12], "don": [11, 15], "framer": 11, "correct": 11, "just": 11, "while": 11, "still": [11, 18], "being": 11, "rmtree": 11, "total": [11, 12, 13, 15, 16, 19], "script": [11, 12, 13, 15], "35": [11, 16, 19], "689": [11, 16, 19], "jupyt": [11, 12, 13, 14, 15], "notebook": [11, 12, 13, 14, 15], "approximate_mod": [11, 16, 19], "ipynb": [11, 12, 13, 15], "code": [11, 12, 13, 14, 15], "py": [11, 12, 13, 15, 16, 19], "zip": [11, 12, 13, 14, 15], "galleri": [11, 12, 13, 14, 15, 19], "sphinx": [11, 12, 13, 14, 15], "nvidia": 12, "hardwar": 12, "matrix": 12, "here": 12, "kernel": 12, "respect": [12, 13], "decompress": 12, "rgb": 12, "subsequ": 12, "scale": 12, "crop": 12, "rotat": 12, "leav": 12, "memori": 12, "fetch": 12, "main": [12, 15], "befor": [12, 15], "packet": 12, "often": [12, 15], "much": [12, 15], "smaller": [12, 15], "pci": 12, "bandwidth": 12, "over": [12, 15, 17], "few": [12, 15], "scenario": 12, "larg": 12, "resolut": 12, "satur": 12, "want": [12, 15, 18], "whole": 12, "imag": 12, "convolut": 12, "after": [12, 15], "free": 12, "other": [12, 13, 15], "work": 12, "situat": 12, "sens": 12, "compar": [12, 15], "small": 12, "transfer": 12, "latenc": 12, "alreadi": [12, 13, 15, 18], "busi": 12, "experi": 12, "improv": 12, "guid": 12, "readm": [12, 18], "librari": [12, 18], "compil": 12, "__version__": 12, "is_avail": 12, "get_device_properti": 12, "dev20250205": 12, "cu126": 12, "_cudadeviceproperti": 12, "name": 12, "tesla": 12, "m60": 12, "major": 12, "minor": 12, "total_memori": 12, "7606mb": 12, "multi_processor_count": 12, "16": 12, "uuid": 12, "de4a564a": 12, "4a37": 12, "62ed": 12, "10ac": 12, "6e1818c37313": 12, "l2_cache_s": 12, "2mb": 12, "264": 12, "960x540": 12, "29": 12, "pixel": 12, "yuv420p": 12, "urllib": 12, "video_fil": 12, "urlretriev": 12, "torchaudio": 12, "asset": 12, "nasas_most_scientifically_complex_space_observatory_requires_precis": 12, "mp4_small": 12, "client": 12, "httpmessag": 12, "0x7f47267d9700": 12, "To": 12, "need": [12, 15], "dtype": [12, 13, 15], "540": 12, "960": 12, "look": [12, 15], "them": 12, "12": 12, "19": 12, "45": 12, "131": 12, "180": 12, "cpu_decod": 12, "cuda_decod": 12, "cpu_fram": 12, "cuda_fram": 12, "plot_cpu_and_cuda_fram": 12, "matplotlib": [12, 13, 15], "pyplot": [12, 13, 15], "plt": [12, 13, 15], "v2": [12, 13, 15], "function": [12, 13, 15], "to_pil_imag": [12, 13, 15], "except": [12, 13, 15], "importerror": [12, 13, 15], "cannot": [12, 13, 15], "plot": [12, 13, 15], "pip": [12, 13, 15], "n_row": 12, "fig": [12, 13, 15], "ax": [12, 13, 15], "subplot": [12, 13, 15], "figsiz": 12, "imshow": [12, 13, 15], "set_titl": [12, 13, 15], "fontsiz": 12, "24": 12, "setp": 12, "xtick": [12, 13, 15], "ytick": [12, 13, 15], "tight_layout": [12, 13, 15], "similar": [12, 15], "human": 12, "ey": 12, "subtl": 12, "math": 12, "frames_equ": 12, "mean_abs_diff": 12, "ab": 12, "max_abs_diff": 12, "fals": 12, "5636": 12, "910": [12, 16, 19], "basic_cuda_exampl": [12, 16, 18, 19], "learn": [13, 15], "util": [13, 15], "raw_video_byt": [13, 15], "titl": [13, 15], "make_grid": [13, 15], "instal": [13, 15, 18], "rcparam": [13, 15], "savefig": [13, 15], "bbox": [13, 15], "tight": [13, 15], "xticklabel": [13, 15], "yticklabel": [13, 15], "now": [13, 15], "cours": 13, "input": [13, 15], "rather": 13, "ha": 13, "yet": 13, "been": 13, "via": 13, "first_fram": 13, "every_twenty_fram": 13, "18": 13, "By": [13, 15], "present": [13, 17], "re": [13, 15], "chang": 13, "everi": 13, "normal": 13, "assert": 13, "isinst": 13, "pure": 13, "addit": [13, 15], "inform": 13, "achiev": [13, 18], "_frame": 13, "76": 13, "04": 13, "other_fram": 13, "4000": 13, "0000": [13, 15], "float64": [13, 15], "0400": [13, 15], "both": [13, 15], "far": [13, 15], "frame_at_2_second": 13, "0800": 13, "2800": [13, 15], "092": [13, 16, 19], "basic_exampl": [13, 16, 18, 19], "generated_examples_python": 14, "generated_examples_jupyt": 14, "denot": 15, "model": [15, 18], "familiar": 15, "quick": 15, "our": [15, 18], "simpl": [15, 18], "principl": 15, "rng": 15, "control": 15, "seed": 15, "reproduc": 15, "hard": 15, "train": [15, 18], "manual_se": 15, "4": [15, 16, 19], "3600": 15, "4800": 15, "6000": 15, "7200": 15, "2000": 15, "3200": 15, "4400": 15, "5600": 15, "8000": 15, "9200": 15, "1600": 15, "8400": 15, "9600": 15, "6800": 15, "output": 15, "repres": [15, 17], "Its": 15, "what": [15, 17], "give": 15, "semant": 15, "includ": 15, "fanci": 15, "easi": [15, 18], "filter": 15, "criteria": 15, "abov": 15, "easili": 15, "out": 15, "who": 15, "specif": 15, "clip_start": 15, "clips_starting_after_five_second": 15, "every_other_clip": 15, "natur": 15, "wai": 15, "cover": 15, "later": 15, "under": 15, "categori": 15, "clips_at_regular_indic": 15, "clips_at_random_timestamp": 15, "clips_at_regular_timestamp": 15, "analog": 15, "ones": 15, "Is": 15, "better": 15, "arguabl": 15, "slightli": 15, "simpler": 15, "possibl": 15, "understand": 15, "discret": 15, "constant": 15, "behav": 15, "region": 15, "undersir": 15, "side": 15, "ensur": [15, 18], "uniform": 15, "caracterist": 15, "along": 15, "interest": 15, "parmet": 15, "There": 15, "thing": 15, "keep": 15, "mind": 15, "upper": 15, "bound": 15, "length": 15, "should": [15, 17], "end_of_video": 15, "8s": 15, "tri": 15, "28": 15, "68": 15, "14": 15, "08": 15, "altern": 15, "necessari": 15, "earli": 15, "enough": 15, "rare": 15, "come": 15, "action": 15, "most": [15, 17, 18], "worri": 15, "too": 15, "620": [15, 16, 19], "00": [16, 19], "46": [16, 19], "310": [16, 19], "execut": [16, 19], "generated_exampl": [16, 18], "mem": [16, 19], "mb": [16, 19], "06": [16, 19], "express": 17, "notion": 17, "quot": 17, "doc": 17, "accord": 17, "variou": 17, "heurist": 17, "expect": 17, "purpos": 17, "cheaper": 17, "group": 17, "aim": 18, "fast": 18, "ecosystem": 18, "ml": 18, "turn": 18, "capabl": 18, "through": 18, "mirror": 18, "version": 18, "fmpeg": 18, "matur": 18, "broad": 18, "coverag": 18, "avail": 18, "system": 18, "abstract": 18, "complex": 18, "correctli": 18, "readi": 18, "fed": 18, "directli": 18, "instruct": 18, "github": 18, "tab": 18, "ov": 18, "demonstr": 18, "develop": 18, "stage": 18, "activ": 18, "feedback": 18, "ani": 18, "suggest": 18, "issu": 18, "repositori": 18}, "objects": {"torchcodec": [[3, 0, 1, "", "Frame"], [4, 0, 1, "", "FrameBatch"]], "torchcodec.Frame": [[3, 1, 1, "", "data"], [3, 1, 1, "", "duration_seconds"], [3, 1, 1, "", "pts_seconds"]], "torchcodec.FrameBatch": [[4, 1, 1, "", "data"], [4, 1, 1, "", "duration_seconds"], [4, 1, 1, "", "pts_seconds"]], "torchcodec.decoders": [[5, 0, 1, "", "VideoDecoder"], [6, 0, 1, "", "VideoStreamMetadata"]], "torchcodec.decoders.VideoDecoder": [[5, 2, 1, "", "__getitem__"], [5, 2, 1, "", "get_frame_at"], [5, 2, 1, "", "get_frame_played_at"], [5, 2, 1, "", "get_frames_at"], [5, 2, 1, "", "get_frames_in_range"], [5, 2, 1, "", "get_frames_played_at"], [5, 2, 1, "", "get_frames_played_in_range"]], "torchcodec.decoders.VideoStreamMetadata": [[6, 3, 1, "", "average_fps"], [6, 1, 1, "", "average_fps_from_header"], [6, 3, 1, "", "begin_stream_seconds"], [6, 1, 1, "", "begin_stream_seconds_from_content"], [6, 1, 1, "", "bit_rate"], [6, 1, 1, "", "codec"], [6, 3, 1, "", "duration_seconds"], [6, 1, 1, "", "duration_seconds_from_header"], [6, 3, 1, "", "end_stream_seconds"], [6, 1, 1, "", "end_stream_seconds_from_content"], [6, 1, 1, "", "height"], [6, 3, 1, "", "num_frames"], [6, 1, 1, "", "num_frames_from_content"], [6, 1, 1, "", "num_frames_from_header"], [6, 1, 1, "", "stream_index"], [6, 1, 1, "", "width"]], "torchcodec.samplers": [[7, 4, 1, "", "clips_at_random_indices"], [8, 4, 1, "", "clips_at_random_timestamps"], [9, 4, 1, "", "clips_at_regular_indices"], [10, 4, 1, "", "clips_at_regular_timestamps"]]}, "objtypes": {"0": "py:class", "1": "py:attribute", "2": "py:method", "3": "py:property", "4": "py:function"}, "objnames": {"0": ["py", "class", "Python class"], "1": ["py", "attribute", "Python attribute"], "2": ["py", "method", "Python method"], "3": ["py", "property", "Python property"], "4": ["py", "function", "Python function"]}, "titleterms": {"torchcodec": [0, 1, 2, 12, 18], "decod": [0, 11, 12, 13, 15], "sampler": [1, 15], "frame": [3, 11, 12, 13], "framebatch": 4, "videodecod": [5, 11, 12, 13], "videostreammetadata": 6, "clips_at_random_indic": 7, "clips_at_random_timestamp": 8, "clips_at_regular_indic": 9, "clips_at_regular_timestamp": 10, "exact": 11, "vs": 11, "approxim": 11, "seek": 11, "mode": 11, "perform": 11, "accuraci": 11, "comparison": 11, "creation": 11, "clip": [11, 15], "sampl": [11, 15], "metadata": 11, "retriev": [11, 13], "what": 11, "thi": 11, "do": 11, "under": 11, "hood": 11, "which": 11, "should": 11, "i": 11, "us": [11, 12, 13], "acceler": 12, "video": [12, 13, 15], "gpu": 12, "cuda": 12, "nvdec": 12, "when": 12, "instal": 12, "enabl": 12, "check": 12, "pytorch": 12, "ha": 12, "download": 12, "visual": 12, "creat": [13, 15], "index": [13, 15], "iter": 13, "over": 13, "pt": 13, "durat": 13, "time": [13, 15, 16, 19], "base": [13, 15], "interact": 14, "exampl": 14, "how": 15, "basic": 15, "manipul": 15, "advanc": 15, "paramet": 15, "rang": 15, "polici": 15, "comput": [16, 19], "glossari": 17, "welcom": 18, "document": 18}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinx.ext.todo": 2, "sphinx.ext.viewcode": 1, "sphinx": 56}})