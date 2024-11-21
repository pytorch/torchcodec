Search.setIndex({"docnames": ["api_ref_decoders", "api_ref_samplers", "api_ref_torchcodec", "generated/torchcodec.Frame", "generated/torchcodec.FrameBatch", "generated/torchcodec.decoders.VideoDecoder", "generated/torchcodec.decoders.VideoStreamMetadata", "generated/torchcodec.samplers.clips_at_random_indices", "generated/torchcodec.samplers.clips_at_random_timestamps", "generated/torchcodec.samplers.clips_at_regular_indices", "generated/torchcodec.samplers.clips_at_regular_timestamps", "generated_examples/basic_cuda_example", "generated_examples/basic_example", "generated_examples/index", "generated_examples/sampling", "generated_examples/sg_execution_times", "glossary", "index", "sg_execution_times"], "filenames": ["api_ref_decoders.rst", "api_ref_samplers.rst", "api_ref_torchcodec.rst", "generated/torchcodec.Frame.rst", "generated/torchcodec.FrameBatch.rst", "generated/torchcodec.decoders.VideoDecoder.rst", "generated/torchcodec.decoders.VideoStreamMetadata.rst", "generated/torchcodec.samplers.clips_at_random_indices.rst", "generated/torchcodec.samplers.clips_at_random_timestamps.rst", "generated/torchcodec.samplers.clips_at_regular_indices.rst", "generated/torchcodec.samplers.clips_at_regular_timestamps.rst", "generated_examples/basic_cuda_example.rst", "generated_examples/basic_example.rst", "generated_examples/index.rst", "generated_examples/sampling.rst", "generated_examples/sg_execution_times.rst", "glossary.rst", "index.rst", "sg_execution_times.rst"], "titles": ["torchcodec.decoders", "torchcodec.samplers", "torchcodec", "Frame", "FrameBatch", "VideoDecoder", "VideoStreamMetadata", "clips_at_random_indices", "clips_at_random_timestamps", "clips_at_regular_indices", "clips_at_regular_timestamps", "Accelerated video decoding on GPUs with CUDA and NVDEC", "Decoding a video with VideoDecoder", "Interactive examples", "How to sample video clips", "Computation times", "Glossary", "Welcome to the TorchCodec documentation!", "Computation times"], "terms": {"For": [0, 1, 7, 9, 10, 14], "tutori": [0, 1, 11, 14], "see": [0, 1, 11, 14], "video": [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 13, 15, 16, 17, 18], "videodecod": [0, 3, 4, 6, 7, 8, 9, 10, 13, 14, 15, 18], "how": [1, 4, 5, 7, 8, 9, 10, 12, 13, 15, 17, 18], "sampl": [1, 4, 5, 7, 8, 9, 10, 13, 15, 17, 18], "clip": [1, 4, 5, 7, 8, 9, 10, 13, 15, 16, 17, 18], "class": [3, 4, 5, 6, 11, 12], "torchcodec": [3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16], "data": [3, 4, 5, 7, 8, 9, 10, 11, 12, 14, 17], "tensor": [3, 4, 5, 11, 12, 14, 17], "pts_second": [3, 4, 7, 8, 9, 10, 12, 14], "float": [3, 4, 5, 6, 8, 10, 11, 14], "duration_second": [3, 4, 6, 7, 8, 9, 10, 12, 14], "sourc": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], "A": [3, 5, 14, 16, 17], "singl": [3, 5, 6, 12], "associ": [3, 4], "metadata": [3, 4, 5, 6, 12, 14, 16], "exampl": [3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 17, 18], "us": [3, 4, 5, 6, 7, 8, 9, 10, 14, 17], "decod": [3, 4, 5, 6, 7, 8, 9, 10, 13, 15, 16, 17, 18], "The": [3, 4, 5, 7, 8, 9, 10, 11, 12, 14, 16], "3": [3, 11, 12, 14, 15, 18], "d": [3, 12], "torch": [3, 4, 5, 11, 12, 14], "durat": [3, 4, 6, 8, 10, 14], "second": [3, 4, 5, 6, 8, 10, 11, 12, 14, 16], "pt": [3, 4, 5, 6, 16], "multipl": [4, 5], "frame": [4, 5, 6, 7, 8, 9, 10, 14, 16], "typic": [4, 14], "4d": [4, 16], "sequenc": [4, 14, 16], "nhwc": [4, 5], "nchw": [4, 5, 11], "5d": [4, 7, 8, 9, 10, 14, 16], "return": [4, 5, 7, 8, 9, 10, 11, 12, 14, 16, 17], "sampler": [4, 7, 8, 9, 10, 16], "when": [4, 5, 7, 8, 9, 10, 12, 14], "resp": 4, "ar": [4, 5, 7, 8, 9, 10, 11, 12, 14, 16, 17], "1d": 4, "2d": [4, 14], "alwai": [4, 5, 12, 14], "cpu": [4, 5, 11, 17], "even": 4, "gpu": [4, 5, 13, 15, 17, 18], "uint8": [4, 11, 12, 14], "union": 5, "str": [5, 6, 7, 8, 9, 10, 12, 14], "path": [5, 12, 14], "byte": [5, 12], "stream_index": [5, 6, 12], "option": [5, 6, 7, 8, 9, 10, 12, 14], "int": [5, 6, 7, 8, 9, 10, 12], "none": [5, 6, 7, 8, 9, 10, 12, 14], "dimension_ord": [5, 7, 8, 9, 10, 12, 14], "liter": [5, 7, 8, 9, 10], "num_ffmpeg_thread": 5, "1": [5, 6, 7, 8, 9, 10, 11, 12, 14], "devic": [5, 11], "stream": [5, 6, 11, 16], "thi": [5, 6, 7, 8, 9, 10, 11, 12, 14], "perform": [5, 6, 14], "scan": [5, 6, 16], "paramet": [5, 7, 8, 9, 10, 11, 12], "pathlib": 5, "If": [5, 6, 7, 8, 9, 10, 14, 17], "local": [5, 12, 14], "file": [5, 12, 14, 15, 16, 17, 18], "object": [5, 6, 11, 12, 14], "raw": [5, 12], "encod": [5, 11, 12], "specifi": 5, "which": [5, 7, 8, 9, 10, 11, 12, 14, 16], "from": [5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 17, 18], "note": [5, 6, 7, 8, 9, 10, 14], "index": [5, 6, 7, 9], "absolut": 5, "across": [5, 6, 14], "all": [5, 6, 13, 14, 18], "media": 5, "type": [5, 7, 8, 9, 10, 12, 14], "left": [5, 11], "unspecifi": 5, "best": [5, 6, 7, 8, 9, 10, 11, 16], "dimens": [5, 12, 14], "order": [5, 12, 16], "can": [5, 11, 12, 14], "either": [5, 14], "default": [5, 7, 8, 9, 10, 12, 14], "where": [5, 7, 8, 9, 10, 11, 12, 14], "n": [5, 12], "batch": [5, 11, 12, 14], "size": [5, 11, 12, 14], "c": [5, 7, 8, 9, 10, 12, 14], "number": [5, 6, 7, 8, 9, 10, 12], "channel": [5, 12], "h": [5, 7, 8, 9, 10, 11, 12, 14], "height": [5, 6, 12], "w": [5, 7, 8, 9, 10, 12, 14], "width": [5, 6, 12], "nativ": [5, 14], "format": [5, 11], "underli": 5, "ffmpeg": [5, 11, 16, 17], "implement": 5, "convert": [5, 11], "those": [5, 7, 8, 9, 10, 14, 17], "cheap": 5, "copi": 5, "oper": 5, "allow": [5, 14], "transform": [5, 11, 12, 14, 17], "torchvis": [5, 10, 11, 12, 14], "http": [5, 11, 12, 14, 17], "pytorch": [5, 14, 17], "org": [5, 11], "vision": 5, "stabl": 5, "html": [5, 17], "_": 5, "thread": 5, "mai": [5, 7, 8, 9, 10, 11, 14, 16], "you": [5, 10, 11, 12, 14, 17], "run": [5, 11, 12, 14], "instanc": [5, 7, 8, 9, 10], "parallel": 5, "higher": 5, "multi": 5, "pass": [5, 11, 12, 14, 16], "0": [5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18], "let": [5, 10, 11, 14, 17], "decid": 5, "variabl": [5, 14], "videostreammetadata": [5, 12], "retriev": [5, 6, 16], "wa": [5, 6, 7, 8, 9, 10], "provid": 5, "initi": 5, "same": [5, 14], "valu": [5, 6, 7, 8, 9, 10, 12, 14], "acceler": [5, 13, 15, 18], "cuda": [5, 13, 15, 17, 18], "nvdec": [5, 13, 15, 18], "__getitem__": 5, "kei": 5, "integr": [5, 17], "slice": [5, 12], "given": [5, 14], "rang": [5, 7, 8, 9, 10, 11], "s": [5, 6, 11, 14, 17], "get_frame_at": [5, 12], "get_frame_played_at": [5, 6, 12], "plai": [5, 6, 12, 16], "timestamp": [5, 8, 10, 11, 14], "time": [5, 6, 8, 10, 11, 16], "stamp": [5, 12, 16], "get_frames_at": [5, 12], "indic": [5, 7, 9, 14], "list": 5, "framebatch": [5, 7, 8, 9, 10, 12, 14, 16], "call": [5, 6, 11], "method": [5, 12], "more": [5, 6, 8, 10, 12, 14], "effici": [5, 14, 17], "repeat": [5, 7, 8, 9, 10, 14], "individu": 5, "make": [5, 11, 14], "sure": 5, "twice": 5, "also": [5, 11, 12, 14], "avoid": 5, "backward": 5, "seek": [5, 17], "slow": 5, "get_frames_in_rang": 5, "start": [5, 7, 8, 9, 10, 14, 17], "stop": 5, "step": [5, 11], "first": [5, 6, 7, 8, 9, 10, 12, 14], "end": [5, 6, 7, 8, 9, 10, 14], "exclus": [5, 7, 8, 9, 10], "per": [5, 7, 8, 9, 10, 12], "python": [5, 11, 12, 13, 14, 17], "convent": [5, 17], "between": [5, 7, 8, 9, 10], "within": [5, 6, 7, 8, 9, 10, 14], "get_frames_played_at": [5, 11, 12], "get_frames_played_in_rang": 5, "start_second": 5, "stop_second": 5, "half": 5, "open": [5, 10, 14, 17], "each": [5, 8, 10, 14], "insid": 5, "As": 5, "exclud": 5, "duration_seconds_from_head": [6, 12], "bit_rat": [6, 12], "num_frames_from_head": [6, 12], "num_frames_from_cont": [6, 12], "begin_stream_second": [6, 12], "end_stream_second": [6, 12, 14], "codec": [6, 11, 12], "average_fps_from_head": [6, 12], "properti": [6, 11], "average_fp": [6, 8, 10, 12], "averag": [6, 8, 10], "fp": [6, 11, 14], "perfom": 6, "comput": 6, "otherwis": 6, "we": [6, 7, 8, 9, 10, 11, 12, 14, 17], "fall": [6, 14], "back": 6, "aver": 6, "obtain": 6, "header": [6, 12, 14], "recommend": [6, 14], "attribut": [6, 12], "instead": [6, 10], "begin": [6, 7, 8, 9, 10], "conceptu": 6, "correspond": [6, 8, 10, 16], "It": [6, 11, 16, 17], "min": 6, "usual": [6, 16], "equal": [6, 9, 10, 11], "bit": [6, 11, 12, 14], "rate": 6, "try": [6, 11, 12, 14], "calcul": 6, "actual": [6, 11], "could": 6, "inaccur": 6, "last_fram": [6, 12], "max": [6, 11], "so": [6, 7, 8, 9, 10, 11, 12, 14, 16], "would": [6, 7, 8, 9, 10, 14], "result": [6, 8, 10, 11], "an": [6, 7, 8, 9, 10, 14, 16, 17], "error": [6, 7, 8, 9, 10], "last": [6, 7, 8, 9, 10, 12, 14], "done": 6, "simpli": [6, 11, 14], "num_fram": [6, 12], "made": 6, "content": [6, 12, 14], "doesn": [6, 11, 14], "t": [6, 11, 14], "involv": [6, 16], "accur": [6, 8, 10], "than": [6, 11, 12, 16], "potenti": 6, "num_clip": [7, 8, 9, 10, 14], "num_frames_per_clip": [7, 8, 9, 10, 14], "num_indices_between_fram": [7, 9, 14], "sampling_range_start": [7, 8, 9, 10, 14], "sampling_range_end": [7, 8, 9, 10, 14], "polici": [7, 8, 9, 10], "repeat_last": [7, 8, 9, 10, 14], "wrap": [7, 8, 9, 10, 14], "random": [7, 8, 14, 17], "mean": [7, 9, 11, 14], "consecut": [7, 9, 16], "sometim": [7, 9, 12, 14], "refer": [7, 9, 11], "dilat": [7, 9], "defin": [7, 8, 9, 10, 12, 14], "i": [7, 8, 9, 10, 11], "e": [7, 8, 9, 10, 11], "onli": [7, 8, 9, 10, 12, 14], "set": [7, 8, 9, 10, 12, 14], "automat": [7, 8, 9, 10, 14], "never": [7, 8, 9, 10], "span": [7, 8, 9, 10], "beyond": [7, 8, 9, 10, 14], "valid": [7, 8, 9, 10, 14], "99": [7, 9], "10": [7, 8, 9, 10, 12, 14, 15, 18], "90": [7, 9], "neg": [7, 8, 9, 10], "accept": [7, 9], "equival": [7, 9, 11], "len": [7, 9, 11, 12], "val": [7, 9], "construct": [7, 8, 9, 10, 14], "describ": [7, 8, 9, 10], "assum": [7, 8, 9, 10], "95": [7, 9], "5": [7, 8, 9, 10, 11, 12, 14], "2": [7, 8, 9, 10, 11, 12, 14], "suppos": [7, 8, 9, 10], "97": [7, 9, 11], "101": [7, 9], "103": [7, 9, 12, 15, 18], "But": [7, 8, 9, 10], "invalid": [7, 8, 9, 10, 14], "replac": [7, 8, 9, 10, 14], "get": [7, 8, 9, 10, 12, 14, 17], "around": [7, 8, 9, 10, 14], "rais": [7, 8, 9, 10, 12, 14], "unlik": [7, 8, 9, 10], "relev": [7, 8, 9, 10], "shape": [7, 8, 9, 10, 11, 12, 14], "field": [7, 8, 9, 10, 12, 14], "depend": [7, 8, 9, 10, 14], "seconds_between_fram": [8, 10, 14], "point": [8, 10], "becaus": [8, 10, 11, 14], "interv": [8, 10, 14], "exactli": [8, 10, 14], "space": [8, 9, 10], "thei": [8, 10, 11, 12, 14], "some": [8, 10, 12, 14], "why": [8, 10], "end_video_second": [8, 10], "seekabl": [8, 10], "9": [8, 10, 14], "7": [8, 10, 11], "11": [8, 10, 14], "regular": [9, 10, 17], "seconds_between_clip_start": [10, 14], "consist": 10, "exist": 10, "api": [10, 11, 14, 17], "take": [10, 14], "find": 10, "support": [10, 11, 14], "pleas": [10, 11, 12, 14, 17], "know": [10, 17], "featur": 10, "request": [10, 11, 12, 14], "nvidia": 11, "hardwar": 11, "matrix": 11, "here": 11, "speed": [11, 14], "up": 11, "kernel": 11, "respect": [11, 12], "decompress": 11, "rgb": 11, "faster": 11, "subsequ": 11, "like": [11, 12, 16], "scale": 11, "crop": 11, "rotat": 11, "leav": 11, "memori": 11, "have": [11, 12, 14, 17], "fetch": 11, "main": [11, 14], "befor": [11, 14], "packet": 11, "often": [11, 14], "much": [11, 14], "smaller": [11, 14], "less": 11, "pci": 11, "bandwidth": 11, "offer": [11, 14], "over": [11, 14, 16], "few": [11, 14], "scenario": 11, "larg": 11, "resolut": 11, "satur": 11, "want": [11, 14, 17], "do": [11, 17], "whole": 11, "imag": 11, "convolut": 11, "after": [11, 14], "your": 11, "free": 11, "other": [11, 12, 14], "work": 11, "situat": 11, "sens": 11, "exact": 11, "compar": [11, 14], "small": 11, "transfer": 11, "latenc": 11, "alreadi": [11, 12, 14, 17], "busi": 11, "experi": 11, "improv": 11, "case": [11, 14], "With": [11, 14], "guid": 11, "readm": [11, 17], "requir": 11, "librari": [11, 17], "compil": 11, "import": [11, 12, 14], "print": [11, 12, 14], "f": [11, 12, 14], "__version__": 11, "is_avail": 11, "get_device_properti": 11, "cu124": 11, "true": 11, "_cudadeviceproperti": 11, "name": 11, "tesla": 11, "m60": 11, "major": 11, "minor": 11, "total_memori": 11, "7606mb": 11, "multi_processor_count": 11, "16": 11, "uuid": 11, "3d6323ee": 11, "fe08": 11, "6085": 11, "003e": 11, "6d411ddcb016": 11, "l2_cache_s": 11, "2mb": 11, "follow": [11, 14], "264": 11, "960x540": 11, "29": 11, "pixel": 11, "yuv420p": 11, "urllib": 11, "video_fil": 11, "mp4": [11, 12, 14], "urlretriev": 11, "torchaudio": 11, "asset": 11, "nasas_most_scientifically_complex_space_observatory_requires_precis": 11, "mp4_small": 11, "client": 11, "httpmessag": 11, "0x7ff53610ce20": 11, "To": 11, "need": [11, 14], "dtype": [11, 12, 14], "540": 11, "960": 11, "look": [11, 14], "them": 11, "against": 11, "12": 11, "19": 11, "45": 11, "131": 11, "180": 11, "cpu_decod": 11, "cuda_decod": 11, "cpu_fram": 11, "cuda_fram": 11, "def": [11, 12, 14], "plot_cpu_and_cuda_fram": 11, "matplotlib": [11, 12, 14], "pyplot": [11, 12, 14], "plt": [11, 12, 14], "v2": [11, 12, 14], "function": [11, 12, 14], "to_pil_imag": [11, 12, 14], "except": [11, 12, 14], "importerror": [11, 12, 14], "cannot": [11, 12, 14], "plot": [11, 12, 14], "pip": [11, 12, 14], "n_row": 11, "fig": [11, 12, 14], "ax": [11, 12, 14], "subplot": [11, 12, 14], "figsiz": 11, "8": [11, 12, 14], "imshow": [11, 12, 14], "set_titl": [11, 12, 14], "fontsiz": 11, "24": 11, "setp": 11, "xtick": [11, 12, 14], "ytick": [11, 12, 14], "tight_layout": [11, 12, 14], "similar": [11, 14], "human": 11, "ey": 11, "subtl": 11, "differ": [11, 14, 16], "math": 11, "frames_equ": 11, "mean_abs_diff": 11, "ab": 11, "max_abs_diff": 11, "fals": 11, "8629": 11, "total": [11, 12, 14, 15, 18], "script": [11, 12, 14], "minut": [11, 12, 14], "6": 11, "895": [11, 15, 18], "jupyt": [11, 12, 13, 14], "notebook": [11, 12, 13, 14], "basic_cuda_exampl": [11, 15, 17, 18], "ipynb": [11, 12, 14], "code": [11, 12, 13, 14], "py": [11, 12, 14, 15, 18], "zip": [11, 12, 13, 14], "galleri": [11, 12, 13, 14, 18], "gener": [11, 12, 13, 14], "sphinx": [11, 12, 13, 14], "In": [12, 14, 16], "ll": [12, 14], "learn": [12, 14], "boilerpl": [12, 14], "download": [12, 13, 14], "web": [12, 14], "util": [12, 14], "ignor": [12, 14], "part": [12, 14], "jump": [12, 14], "right": [12, 14], "below": [12, 14], "www": [12, 14], "pexel": [12, 14], "com": [12, 14, 17], "dog": [12, 14], "eat": [12, 14], "854132": [12, 14], "licens": [12, 14], "cc0": [12, 14], "author": [12, 14], "coverr": [12, 14], "url": [12, 14], "sd_640_360_25fp": [12, 14], "respons": [12, 14], "user": [12, 14, 16], "agent": [12, 14], "status_cod": [12, 14], "200": [12, 14], "runtimeerror": [12, 14], "fail": [12, 14], "raw_video_byt": [12, 14], "titl": [12, 14], "make_grid": [12, 14], "instal": [12, 14, 17], "rcparam": [12, 14], "savefig": [12, 14], "bbox": [12, 14], "tight": [12, 14], "xticklabel": [12, 14], "yticklabel": [12, 14], "now": [12, 14], "cours": 12, "input": [12, 14], "rather": 12, "ha": 12, "yet": 12, "been": 12, "access": 12, "via": 12, "345": 12, "13": [12, 14], "25": 12, "505790": 12, "h264": 12, "640": [12, 14], "360": [12, 14], "first_fram": 12, "every_twenty_fram": 12, "20": 12, "18": 12, "By": [12, 14], "present": [12, 16], "re": [12, 14], "one": 12, "chang": 12, "everi": 12, "normal": 12, "assert": 12, "isinst": 12, "pure": 12, "addit": [12, 14], "inform": 12, "about": [12, 14, 16], "achiev": [12, 17], "_frame": 12, "76": 12, "04": 12, "other_fram": 12, "50": 12, "4000": 12, "0000": [12, 14], "float64": [12, 14], "0400": [12, 14], "both": [12, 14], "contain": [12, 14], "far": [12, 14], "frame_at_2_second": 12, "0800": 12, "2800": [12, 14], "basic_exampl": [12, 15, 17, 18], "generated_examples_python": 13, "generated_examples_jupyt": 13, "denot": 14, "model": [14, 17], "familiar": 14, "quick": 14, "our": [14, 17], "simpl": [14, 17], "principl": 14, "clips_at_random_indic": 14, "rng": 14, "control": 14, "seed": 14, "reproduc": 14, "hard": 14, "train": [14, 17], "manual_se": 14, "4": 14, "3600": 14, "4800": 14, "6000": 14, "7200": 14, "2000": 14, "3200": 14, "4400": 14, "5600": 14, "8000": 14, "9200": 14, "1600": 14, "8400": 14, "9600": 14, "6800": 14, "output": 14, "repres": [14, 16], "Its": 14, "what": [14, 16], "give": 14, "semant": 14, "includ": 14, "fanci": 14, "easi": [14, 17], "filter": 14, "criteria": 14, "abov": 14, "easili": 14, "out": 14, "who": 14, "specif": 14, "clip_start": 14, "clips_starting_after_five_second": 14, "every_other_clip": 14, "natur": 14, "wai": 14, "reli": [14, 17], "cover": 14, "later": 14, "ve": 14, "under": 14, "two": 14, "categori": 14, "clips_at_regular_indic": 14, "clips_at_random_timestamp": 14, "clips_at_regular_timestamp": 14, "analog": 14, "ones": 14, "term": 14, "Is": 14, "better": 14, "arguabl": 14, "slightli": 14, "simpler": 14, "behavior": 14, "possibl": 14, "understand": 14, "discret": 14, "constant": 14, "behav": 14, "howev": [14, 17], "region": 14, "lead": 14, "undersir": 14, "side": 14, "effect": 14, "ensur": [14, 17], "uniform": 14, "caracterist": 14, "along": 14, "entir": [14, 16], "interest": 14, "parmet": 14, "There": 14, "thing": 14, "keep": 14, "mind": 14, "upper": 14, "bound": 14, "length": 14, "should": [14, 16], "end_of_video": 14, "8s": 14, "tri": 14, "28": 14, "68": 14, "14": 14, "08": 14, "altern": 14, "necessari": 14, "earli": 14, "enough": 14, "rare": 14, "come": 14, "action": 14, "most": [14, 16, 17], "probabl": 14, "don": 14, "worri": 14, "too": 14, "583": [14, 15, 18], "00": [15, 18], "581": [15, 18], "execut": [15, 18], "generated_exampl": [15, 17], "mem": [15, 18], "mb": [15, 18], "06": [15, 18], "03": [15, 18], "express": 16, "notion": 16, "determin": 16, "quot": 16, "doc": 16, "accord": 16, "variou": 16, "heurist": 16, "expect": 16, "purpos": 16, "doe": 16, "lot": 16, "cheaper": 16, "necessarili": 16, "group": 16, "aim": 17, "fast": 17, "well": 17, "ecosystem": 17, "ml": 17, "turn": 17, "capabl": 17, "through": 17, "mirror": 17, "version": 17, "fmpeg": 17, "matur": 17, "broad": 17, "coverag": 17, "avail": 17, "system": 17, "abstract": 17, "complex": 17, "correctli": 17, "readi": 17, "fed": 17, "directli": 17, "instruct": 17, "github": 17, "tab": 17, "ov": 17, "demonstr": 17, "still": 17, "develop": 17, "stage": 17, "activ": 17, "feedback": 17, "ani": 17, "suggest": 17, "issu": 17, "repositori": 17}, "objects": {"torchcodec": [[3, 0, 1, "", "Frame"], [4, 0, 1, "", "FrameBatch"]], "torchcodec.Frame": [[3, 1, 1, "", "data"], [3, 1, 1, "", "duration_seconds"], [3, 1, 1, "", "pts_seconds"]], "torchcodec.FrameBatch": [[4, 1, 1, "", "data"], [4, 1, 1, "", "duration_seconds"], [4, 1, 1, "", "pts_seconds"]], "torchcodec.decoders": [[5, 0, 1, "", "VideoDecoder"], [6, 0, 1, "", "VideoStreamMetadata"]], "torchcodec.decoders.VideoDecoder": [[5, 2, 1, "", "__getitem__"], [5, 2, 1, "", "get_frame_at"], [5, 2, 1, "", "get_frame_played_at"], [5, 2, 1, "", "get_frames_at"], [5, 2, 1, "", "get_frames_in_range"], [5, 2, 1, "", "get_frames_played_at"], [5, 2, 1, "", "get_frames_played_in_range"]], "torchcodec.decoders.VideoStreamMetadata": [[6, 3, 1, "", "average_fps"], [6, 1, 1, "", "average_fps_from_header"], [6, 1, 1, "", "begin_stream_seconds"], [6, 1, 1, "", "bit_rate"], [6, 1, 1, "", "codec"], [6, 3, 1, "", "duration_seconds"], [6, 1, 1, "", "duration_seconds_from_header"], [6, 1, 1, "", "end_stream_seconds"], [6, 1, 1, "", "height"], [6, 3, 1, "", "num_frames"], [6, 1, 1, "", "num_frames_from_content"], [6, 1, 1, "", "num_frames_from_header"], [6, 1, 1, "", "stream_index"], [6, 1, 1, "", "width"]], "torchcodec.samplers": [[7, 4, 1, "", "clips_at_random_indices"], [8, 4, 1, "", "clips_at_random_timestamps"], [9, 4, 1, "", "clips_at_regular_indices"], [10, 4, 1, "", "clips_at_regular_timestamps"]]}, "objtypes": {"0": "py:class", "1": "py:attribute", "2": "py:method", "3": "py:property", "4": "py:function"}, "objnames": {"0": ["py", "class", "Python class"], "1": ["py", "attribute", "Python attribute"], "2": ["py", "method", "Python method"], "3": ["py", "property", "Python property"], "4": ["py", "function", "Python function"]}, "titleterms": {"torchcodec": [0, 1, 2, 11, 17], "decod": [0, 11, 12, 14], "sampler": [1, 14], "frame": [3, 11, 12], "framebatch": 4, "videodecod": [5, 11, 12], "videostreammetadata": 6, "clips_at_random_indic": 7, "clips_at_random_timestamp": 8, "clips_at_regular_indic": 9, "clips_at_regular_timestamp": 10, "acceler": 11, "video": [11, 12, 14], "gpu": 11, "cuda": 11, "nvdec": 11, "when": 11, "us": [11, 12], "instal": 11, "enabl": 11, "check": 11, "pytorch": 11, "ha": 11, "download": 11, "visual": 11, "creat": [12, 14], "index": [12, 14], "iter": 12, "over": 12, "retriev": 12, "pt": 12, "durat": 12, "time": [12, 14, 15, 18], "base": [12, 14], "interact": 13, "exampl": 13, "how": 14, "sampl": 14, "clip": 14, "basic": 14, "manipul": 14, "advanc": 14, "paramet": 14, "rang": 14, "polici": 14, "comput": [15, 18], "glossari": 16, "welcom": 17, "document": 17}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinx.ext.todo": 2, "sphinx.ext.viewcode": 1, "sphinx": 56}})