
        # measurement_start_time = time.time()
        # if measure:
        #     total_vcons += len(valid_vcons)
        #     total_bytes = dir_size_bytes(settings.processing_dir)
        #     for cur_vcon in valid_vcons:
        #         filename = vcon.get_filename(cur_vcon)
        #         total_bytes += size_of_file(filename)
        #         total_duration += audio.get_duration(filename)
        # measurement_end_time = time.time()
        # measurement_time = measurement_end_time - measurement_start_time
        # start_time += measurement_time

        # load_processing_into_ram_start_time = time.time()
        # print(f"Valid vcons: {len(valid_vcons)}")
        # vcons_in_ram = None
        # with with_timing("Loading valid vcons into RAM."):
        #     vcons_in_ram = vcon.load_processing_into_ram(valid_vcons)
        #     print(f"Vcons in RAM: {len(vcons_in_ram)}")
        # print(f"Vcons in RAM: {vcons_in_ram[0]}")
        # load_processing_into_ram_time = time.time() - load_processing_into_ram_start_time

        # convert_to_mono_start_time = time.time()
        # vcons_mono = None
        # with with_timing("Converting to mono."):
        #     vcons_mono = vcon.convert_to_mono_many(vcons_in_ram)
        #     print(f"Mono vcons: {len(vcons_mono)}")
        # print(f"Mono vcons: {vcons_mono[0]}")
        # convert_to_mono_time = time.time() - convert_to_mono_start_time

        # resample_start_time = time.time()
        # vcons_resampled = None
        # with with_timing("Resampling."):
        #     vcons_resampled = vcon.resample_many(vcons_mono)
        #     print(f"Resampled vcons: {len(vcons_resampled)}")
        # print(f"Resampled vcons: {vcons_resampled[0]}")
        # resample_time = time.time() - resample_start_time

        # apply_vad_start_time = time.time()
        # vcons_vad = None
        # with with_timing("Applying VAD."):
        #     vcons_vad = vcon.apply_vad_many(vcons_resampled)
        #     #vcons_vad = vcons_resampled
        #     print(f"VAD vcons: {len(vcons_vad)}")
        # print(f"VAD vcons: {vcons_vad[0]}")
        # apply_vad_time = time.time() - apply_vad_start_time

        # pad_start_time = time.time()
        # vcons_padded = None
        # with with_timing("Padding."):
        #     vcons_padded = vcon.pad_many(vcons_preprocessed)
        # print(f"Padded vcons: {vcons_padded[0]}")
        # pad_time = time.time() - pad_start_time

        # vcons_on_gpu = None
        # with with_timing("Moving to GPU."):
        #     vcons_on_gpu = vcon.move_to_gpu_many(vcons_vad)
        # print(f"Vcons on GPU: {vcons_on_gpu[0]}")

    batch_start_time = time.time()
    vcons_batched = None
    with with_timing("Batching."):
        vcons_batched = vcon.make_batches(vcons_preprocessed, gpu.batch_bytes())
    batch_time = time.time() - batch_start_time

    vcons_detected = None
    identify_languages_start_time = time.time()
    with with_timing("Identifying languages."):
        # vcons_detected = ai.identify_languages(vcons_batched, lang_detect_model)  # ai module removed
        vcons_detected = vcons_batched  # Placeholder since ai.py is removed
    print(f"Detected vcons.")
    identify_languages_time = time.time() - identify_languages_start_time

    split_by_language_start_time = time.time()
    with with_timing("Splitting by language."):
        vcons_en, vcons_non_en = vcon.split_by_language(vcons_detected)
    # print(f"En vcons: {vcons_en[0]}")
    # print(f"Non-en vcons: {vcons_non_en[0]}")
    split_by_language_time = time.time() - split_by_language_start_time
    
    # batching
    batch_en_start_time = time.time()
    vcons_en_batched = []
    info_header(f"Batching {len(vcons_en)} en vcons.")
    if vcons_en:
        vcons_en_batched = vcon.make_batches(vcons_en, gpu.batch_bytes())
    batch_en_time = time.time() - batch_en_start_time

    batch_non_en_start_time = time.time()
    vcons_non_en_batched = []
    info_header(f"Batching {len(vcons_non_en)} non-en vcons.")
    if vcons_non_en:
        vcons_non_en_batched = vcon.make_batches(vcons_non_en, gpu.batch_bytes())
    batch_non_en_time = time.time() - batch_non_en_start_time

    transcribe_en_start_time = time.time()
    vcons_en_transcribed = []
    with with_timing("Transcribing en vcons."):
        if vcons_en:
            # vcons_en_transcribed = ai.transcribe_many(vcons_en_batched, en_model)  # ai module removed
            vcons_en_transcribed = vcons_en  # Placeholder since ai.py is removed
    transcribe_en_time = time.time() - transcribe_en_start_time

    transcribe_non_en_start_time = time.time()
    vcons_non_en_transcribed = []
    with with_timing("Transcribing non-en vcons."):
        if vcons_non_en:
            # vcons_non_en_transcribed = ai.transcribe_many(vcons_non_en_batched, non_en_model, language="es")  # ai module removed
            vcons_non_en_transcribed = vcons_non_en  # Placeholder since ai.py is removed
    transcribe_non_en_time = time.time() - transcribe_non_en_start_time

    all_vcons = vcons_en_transcribed + vcons_non_en_transcribed

    mark_as_done_start_time = time.time()
    with with_timing("Marking as done."):
        vcon.mark_vcons_as_done(all_vcons)
    mark_as_done_time = time.time() - mark_as_done_start_time

    update_on_db_start_time = time.time()
    with with_timing("Updating on DB."):
        vcon.update_vcons_on_db(all_vcons)
    update_on_db_time = time.time() - update_on_db_start_time

    clear_processing_start_time = time.time()
    with with_timing("Clearing processing."):
        cache.clear_processing()
    clear_processing_time = time.time() - clear_processing_start_time

    program_end_time = time.time()
    program_time = program_end_time - program_start_time

    if measure:
        clear_screen()
        vcons = vcons_en_transcribed + vcons_non_en_transcribed
        print(f"proof of transcription of {len(vcons)} vcons")
        for cur_vcon in vcons:
            print(f"vcon: {cur_vcon["_id"]} languages: {vcon.get_languages(cur_vcon)}")
            print(f"transcript: {vcon.get_transcript_text(cur_vcon)}")
        total_bytes_mb = total_bytes / (1024 * 1024)
        print(f"--------------------------------")
        print(f"Total vcons: {total_vcons}")
        print(f"Total duration: {total_duration}")
        print(f"Total processed: {total_bytes_mb:.2f}MB")
        print(f"RTF: {total_duration / program_time}")
        print(f"Total MB per second: {total_bytes_mb / program_time:.2f}MB/s")
        print(f"--------------------------------")
        # print(f"Move downloading to processing: {move_downloading_to_processing_time:.2f}s {move_downloading_to_processing_time/program_time*100:.2f}%")
        # print(f"Processing invalids: {processing_invalids_time:.2f}s {processing_invalids_time/program_time*100:.2f}%")
        # print(f"Load processing into RAM: {load_processing_into_ram_time:.2f}s {load_processing_into_ram_time/program_time*100:.2f}%")
        # print(f"Convert to mono: {convert_to_mono_time:.2f}s {convert_to_mono_time/program_time*100:.2f}%")
        # print(f"Resample: {resample_time:.2f}s {resample_time/program_time*100:.2f}%")
        # print(f"Apply VAD: {apply_vad_time:.2f}s {apply_vad_time/program_time*100:.2f}%")
        # print(f"Pad: {pad_time:.2f}s {pad_time/program_time*100:.2f}%")
        print(f"Preprocess: {preprocess_time:.2f}s {preprocess_time/program_time*100:.2f}%")
        print(f"Batch: {batch_time:.2f}s {batch_time/program_time*100:.2f}%")
        print(f"Identify languages: {identify_languages_time:.2f}s {identify_languages_time/program_time*100:.2f}%")
        num_en = 0
        num_es = 0
        for vcon_cur in all_vcons:
            languages = vcon.get_languages(vcon_cur)
            if "en" in languages:
                num_en += 1
            if "es" in languages:
                num_es += 1
        print(f"Num en: {num_en} ({num_en/len(all_vcons)*100:.2f}%)")
        print(f"Num es: {num_es} ({num_es/len(all_vcons)*100:.2f}%)")
        print(f"Split by language: {split_by_language_time:.2f}s {split_by_language_time/program_time*100:.2f}%")
        print(f"Batch en: {batch_en_time:.2f}s {batch_en_time/program_time*100:.2f}%")
        print(f"Batch non-en: {batch_non_en_time:.2f}s {batch_non_en_time/program_time*100:.2f}%")
        print(f"Transcribe en: {transcribe_en_time:.2f}s {transcribe_en_time/program_time*100:.2f}%")
        print(f"Transcribe non-en: {transcribe_non_en_time:.2f}s {transcribe_non_en_time/program_time*100:.2f}%")
        print(f"Mark as done: {mark_as_done_time:.2f}s {mark_as_done_time/program_time*100:.2f}%")
        print(f"Update on DB: {update_on_db_time:.2f}s {update_on_db_time/program_time*100:.2f}%")
        print(f"Clear processing: {clear_processing_time:.2f}s {clear_processing_time/program_time*100:.2f}%")
        print(f"Max GPU ram utilization: {gpu.max_gpu_memory_usage()/(1024**3):.2f}GB")
        gpu.print_gpu_memory_usage()
        print(f"--------------------------------")
        print(f"Total time: {program_time:.2f}s {program_time/program_time*100:.2f}%")
        keep_running.clear()
        reserver_thread.join()
        return