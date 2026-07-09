    # ======================== Вкладка Калибровка голоса ========================

    def _setup_calibrate_tab(self):
        """Create voice calibration tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)
        layout.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        instr_text = "Calibration Guide:\n\n"
        instr_text += "Add multiple .pt voice vectors (e.g. sad, happy, whisper)\n"
        instr_text += "and adjust their weight. Result = blended custom voice.\n\n"
        instr_text += "How it works:\n"
        instr_text += "  1. Add .pt files of voices with different emotions\n"
        instr_text += "  2. Set weight for each (1.0 = full, 0.5 = half)\n"
        instr_text += "  3. Press Calibrate to get a mixed vector\n"
        instr_text += "  4. Save as a new .pt file\n"
        instr_text += "  5. Use it as a regular voice vector"
        instr_label = QLabel(instr_text)
        instr_label.setStyleSheet("color: #e0e0e0; font-size: 12px; background-color: #0f3460; border-radius: 6px; padding: 10px;")
        instr_label.setWordWrap(True)
        layout.addWidget(instr_label)

        voices_group = QGroupBox("Voice Vectors")
        voices_layout = QVBoxLayout(voices_group)
        self.calibrate_voices_list = QTableWidget()
        self.calibrate_voices_list.setColumnCount(4)
        self.calibrate_voices_list.setHorizontalHeaderLabels([".pt File Path", "Weight", "Duration", ""])
        self.calibrate_voices_list.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.calibrate_voices_list.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.calibrate_voices_list.setColumnWidth(1, 100)
        self.calibrate_voices_list.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.calibrate_voices_list.setColumnWidth(2, 120)
        self.calibrate_voices_list.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self.calibrate_voices_list.setColumnWidth(3, 40)
        self.calibrate_voices_list.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.calibrate_voices_list.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.calibrate_voices_list.setMinimumHeight(200)
        voices_layout.addWidget(self.calibrate_voices_list)

        btn_layout = QHBoxLayout()
        btn_add_voice = QPushButton("Add .pt")
        btn_add_voice.setToolTip("Select a .pt voice vector file from the file browser.")
        btn_add_voice.clicked.connect(self._add_calibrate_voice)
        btn_layout.addWidget(btn_add_voice)
        self.btn_remove_voice = QPushButton("Remove")
        self.btn_remove_voice.setToolTip("Remove the selected voice from the calibration list.")
        self.btn_remove_voice.clicked.connect(self._remove_calibrate_voice)
        self.btn_remove_voice.setStyleSheet("QPushButton { background-color: #3d3d5c; color: #e0e0e0; border-radius: 6px; } QPushButton:hover { background-color: #ef4444; color: white; }")
        btn_layout.addWidget(self.btn_remove_voice)
        btn_load_voice = QPushButton("Refresh")
        btn_load_voice.setToolTip("Reload vector info (duration, size).")
        btn_load_voice.clicked.connect(self._load_calibrate_voice_info)
        btn_layout.addWidget(btn_load_voice)
        btn_layout.addStretch()
        voices_layout.addLayout(btn_layout)
        layout.addWidget(voices_group)

        preset_group = QGroupBox("Quick Presets")
        preset_layout = QHBoxLayout(preset_group)
        preset_label = QLabel("Quick weight:")
        preset_label.setStyleSheet("color: #e0e0e0; font-size: 12px;")
        preset_layout.addWidget(preset_label)
        self.calibrate_preset_combo = QComboBox()
        self.calibrate_preset_combo.addItems(["Equal", "Sad", "Happy", "Angry", "Whisper", "Custom"])
        self.calibrate_preset_combo.setMinimumWidth(120)
        preset_layout.addWidget(self.calibrate_preset_combo)
        self.calibrate_apply_preset_btn = QPushButton("Apply")
        self.calibrate_apply_preset_btn.setToolTip("Apply the selected preset to all voices in the list.")
        self.calibrate_apply_preset_btn.clicked.connect(self._apply_preset)
        self.calibrate_apply_preset_btn.setStyleSheet("QPushButton { background-color: #533483; color: white; border-radius: 6px; } QPushButton:hover { background-color: #7c3aed; }")
        preset_layout.addWidget(self.calibrate_apply_preset_btn)
        preset_layout.addStretch()
        layout.addWidget(preset_group)

        name_group = QGroupBox("Calibrated Voice Name")
        name_layout = QHBoxLayout(name_group)
        self.calibrate_name_edit = QLineEdit()
        self.calibrate_name_edit.setPlaceholderText("e.g. my_custom_voice")
        self.calibrate_name_edit.setMinimumWidth(200)
        self.calibrate_name_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.calibrate_name_edit.setToolTip("Unique name for the calibrated voice.")
        name_layout.addWidget(QLabel("Name:"))
        name_layout.addWidget(self.calibrate_name_edit)
        layout.addWidget(name_group)

        self.btn_calibrate = QPushButton("Calibrate Voice", objectName="calibrateBtn")
        self.btn_calibrate.setObjectName("calibrateBtn")
        self.btn_calibrate.setFixedHeight(48)
        self.btn_calibrate.setMinimumWidth(200)
        self.btn_calibrate.setToolTip("Load vectors, normalize weights, uniform size, weighted sum.")
        self.btn_calibrate.setStyleSheet("QPushButton#calibrateBtn { background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #533483, stop:1 #0f3460); font-size: 15px; font-weight: bold; padding: 10px 24px; }")
        self.btn_calibrate.clicked.connect(self._start_calibrate)
        layout.addWidget(self.btn_calibrate, 0, Qt.AlignmentFlag.AlignCenter)

        self.calibrate_progress = QProgressBar()
        self.calibrate_progress.setVisible(False)
        layout.addWidget(self.calibrate_progress)

        result_group = QGroupBox("Calibration Result")
        result_layout = QVBoxLayout(result_group)
        self.calibrate_result_label = QLabel("No result")
        self.calibrate_result_label.setStyleSheet("color: #888; font-style: italic;")
        result_layout.addWidget(self.calibrate_result_label)
        btn_save_calibrated = QPushButton("Save calibrated .pt")
        btn_save_calibrated.setToolTip("Save the mixed vector to a .pt file.")
        btn_save_calibrated.clicked.connect(self._save_calibrated_vector)
        result_layout.addWidget(btn_save_calibrated)
        btn_test_calibrated = QPushButton("Test on TTS")
        btn_test_calibrated.setToolTip("Save calibrated .pt and switch to TTS tab to test.")
        btn_test_calibrated.clicked.connect(self._test_calibrated_voice)
        btn_test_calibrated.setStyleSheet("QPushButton { background-color: #22c55e; color: white; font-weight: bold; border-radius: 6px; } QPushButton:hover { background-color: #16a34a; }")
        result_layout.addWidget(btn_test_calibrated)
        result_layout.addStretch()
        layout.addWidget(result_group)
        layout.addStretch()
        tab.setLayout(layout)
        self.tabs.addTab(tab, "Voice Calibration")
        self._calibrated_vector = None

    def _add_calibrate_voice(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select .pt Voice Vector", "", "Voice Vector Files (*.pt);;All Files (*)")
        if not path: return
        row = self.calibrate_voices_list.rowCount()
        self.calibrate_voices_list.insertRow(row)
        self.calibrate_voices_list.setItem(row, 0, QTableWidgetItem(path))
        self.calibrate_voices_list.setItem(row, 1, QTableWidgetItem("1.0"))
        self.calibrate_voices_list.setItem(row, 2, QTableWidgetItem("--"))
        del_btn = QPushButton("x")
        del_btn.setFixedWidth(30)
        del_btn.setStyleSheet("QPushButton { background-color: transparent; color: #ef4444; border: none; } QPushButton:hover { background-color: #3d3d5c; }")
        del_btn.clicked.connect(lambda checked, r=row: self.calibrate_voices_list.removeRow(r))
        self.calibrate_voices_list.setCellWidget(row, 3, del_btn)
        self._load_calibrate_voice_info()

    def _remove_calibrate_voice(self):
        current = self.calibrate_voices_list.currentRow()
        if current >= 0: self.calibrate_voices_list.removeRow(current)

    def _load_calibrate_voice_info(self):
        for row in range(self.calibrate_voices_list.rowCount()):
            path_item = self.calibrate_voices_list.item(row, 0)
            if not path_item: continue
            path = path_item.text()
            if not os.path.exists(path):
                self.calibrate_voices_list.setItem(row, 2, QTableWidgetItem("X No"))
                continue
            try:
                vector = torch.load(path, map_location="cpu", weights_only=False)
                if isinstance(vector, dict):
                    xvector = vector.get('xvector', vector.get('voice_embedding', vector.get('embedding', vector)))
                else:
                    xvector = vector
                if isinstance(xvector, torch.Tensor):
                    xvector = xvector.cpu()
                if isinstance(xvector, torch.Tensor):
                    duration_sec = len(xvector) / 1024.0
                    self.calibrate_voices_list.setItem(row, 2, QTableWidgetItem(f"{duration_sec:.1f}s | {xvector.shape}"))
                else:
                    self.calibrate_voices_list.setItem(row, 2, QTableWidgetItem("X"))
            except Exception as e:
                self.calibrate_voices_list.setItem(row, 2, QTableWidgetItem(f"X {str(e)[:15]}"))

    def _apply_preset(self):
        preset = self.calibrate_preset_combo.currentText()
        count = self.calibrate_voices_list.rowCount()
        if count == 0: return
        wm = {"Equal": "1.0", "Sad": "0.7", "Happy": "0.7", "Angry": "0.7", "Whisper": "0.6"}
        weight = wm.get(preset, "1.0")
        if preset == "Custom": return
        for row in range(count):
            self.calibrate_voices_list.setItem(row, 1, QTableWidgetItem(weight))

    def _start_calibrate(self):
        if self.calibrate_voices_list.rowCount() == 0:
            QMessageBox.warning(self, "Attention", "Add at least one .pt voice file.")
            return
        vectors, weights = [], []
        for row in range(self.calibrate_voices_list.rowCount()):
            path_item = self.calibrate_voices_list.item(row, 0)
            weight_item = self.calibrate_voices_list.item(row, 1)
            if not path_item or not weight_item: continue
            path = path_item.text()
            weight = float(weight_item.text())
            if not os.path.exists(path):
                QMessageBox.warning(self, "Error", f"File not found: {path}")
                return
            try:
                data = torch.load(path, map_location="cpu", weights_only=False)
                if isinstance(data, dict):
                    xvector = data.get('xvector', data.get('voice_embedding', data.get('embedding', data)))
                else:
                    xvector = data
                if isinstance(xvector, torch.Tensor):
                    vectors.append(xvector.float())
                else:
                    vectors.append(xvector)
                weights.append(weight)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Cannot read {path}:\n{e}")
                return
        if len(vectors) == 0:
            QMessageBox.warning(self, "Attention", "Cannot load vectors.")
            return
        total_weight = sum(weights)
        if total_weight == 0:
            QMessageBox.warning(self, "Error", "Sum of weights is 0.")
            return
        self.calibrate_progress.setVisible(True)
        self.calibrate_progress.setValue(0)
        self.btn_calibrate.setEnabled(False)
        self.btn_calibrate.setText("Calibrating...")
        self._set_status("Calibrating voice...", "#fbbf24")
        try:
            norm_weights = [w / total_weight for w in weights]
            max_dim = max(v.shape[-1] for v in vectors)
            padded_vectors = []
            for v in vectors:
                if v.shape[-1] < max_dim:
                    padded = torch.zeros(max_dim)
                    padded[:v.shape[-1]] = v
                    padded_vectors.append(padded)
                else:
                    padded_vectors.append(v)
            calibrated = sum(w * v for w, v in zip(norm_weights, padded_vectors))
            calibrated = calibrated / calibrated.norm()
            self._calibrated_vector = calibrated.cpu().numpy()
            self.calibrate_progress.setValue(100)
            self.calibrate_result_label.setText(f"Calibration successful! Vectors: {len(vectors)} | Size: {calibrated.shape}")
            self.calibrate_result_label.setStyleSheet("color: #4ade80; font-weight: bold;")
            self._set_status("Calibration complete", "#4ade80")
        except Exception as e:
            tb = traceback.format_exc()
            self._log(f"Calibration error: {tb}")
            QMessageBox.critical(self, "Error", f"Cannot calibrate:\n{e}")
            self._set_status("Calibration error", "#ef4444")
        finally:
            self.calibrate_progress.setVisible(False)
            self.btn_calibrate.setEnabled(True)
            self.btn_calibrate.setText("Calibrate Voice")

    def _save_calibrated_vector(self):
        if self._calibrated_vector is None:
            QMessageBox.warning(self, "Attention", "Run calibration first.")
            return
        name = self.calibrate_name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Attention", "Enter a name for the calibrated voice.")
            return
        first_path = None
        for row in range(self.calibrate_voices_list.rowCount()):
            item = self.calibrate_voices_list.item(row, 0)
            if item and os.path.exists(item.text()):
                first_path = item.text()
                break
        if first_path:
            default_dir = os.path.dirname(first_path)
        else:
            default_dir = os.path.expanduser("~")
        default_name = f"{name}_calibrated.pt"
        path, _ = QFileDialog.getSaveFileName(self, 'Save Calibrated Vector', default_name, 'Voice Vector Files (*.pt)')
        if not path: return
        try:
            saved_data = {'xvector': torch.tensor(self._calibrated_vector), 'name': name, 'calibrated': True}
            torch.save(saved_data, path)
            QMessageBox.information(self, "Success", f"Saved:\n{path}")
            self._set_status("Saved", "#4ade80")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot save:\n{e}")

    def _test_calibrated_voice(self):
        if self._calibrated_vector is None:
            QMessageBox.warning(self, "Attention", "Run calibration first.")
            return
        name = self.calibrate_name_edit.text().strip() or "calibrated_voice"
        tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_path = os.path.join(tmp_dir, f"_calibrate_test_{name}.pt")
        try:
            torch.save({'xvector': torch.tensor(self._calibrated_vector), 'name': name, 'calibrated': True}, tmp_path)
            self.voice_vector_path.setText(tmp_path)
            self.tabs.setCurrentIndex(0)
            QMessageBox.information(self, "Switch to TTS", f"Voice \"{name}\" ready.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Test error:\n{e}")


    # ======================== Вкладка Дообучение ========================

    def _setup_finetune_tab(self):
        """Create fine-tuning tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)
        layout.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        finetune_instr = "Model Fine-Tuning\n\n"
        finetune_instr += "Fine-tune the Qwen3-TTS model on your own dataset.\n\n"
        finetune_instr += "Requirements:\n"
        finetune_instr += "  * JSONL dataset (audio + text + ref_audio)\n"
        finetune_instr += "  * Qwen3-TTS-12Hz model (base or custom)\n"
        finetune_instr += "  * GPU with >= 12GB VRAM"
        instr_label = QLabel(finetune_instr)
        instr_label.setStyleSheet("color: #e0e0e0; font-size: 12px; background-color: #0f3460; border-radius: 6px; padding: 10px;")
        instr_label.setWordWrap(True)
        layout.addWidget(instr_label)

        model_group = QGroupBox("Model")
        model_layout = QVBoxLayout(model_group)
        self.finetune_model_path_edit = QLineEdit()
        self.finetune_model_path_edit.setPlaceholderText("Path to Qwen3-TTS model")
        self.finetune_model_path_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.finetune_model_path_edit.setToolTip("Folder with the model for fine-tuning.")
        model_layout.addWidget(self.finetune_model_path_edit)
        btn_layout_model = QHBoxLayout()
        btn_browse_model = QPushButton("Select model")
        btn_browse_model.setToolTip("Select Qwen3-TTS model folder.")
        btn_browse_model.clicked.connect(self._browse_finetune_model)
        btn_layout_model.addWidget(btn_browse_model)
        btn_layout_model.addStretch()
        model_layout.addLayout(btn_layout_model)
        layout.addWidget(model_group)

        dataset_group = QGroupBox("Dataset")
        dataset_layout = QVBoxLayout(dataset_group)
        self.finetune_jsonl_edit = QLineEdit()
        self.finetune_jsonl_edit.setPlaceholderText("Path to JSONL dataset file")
        self.finetune_jsonl_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.finetune_jsonl_edit.setToolTip("JSONL file with data for fine-tuning.")
        dataset_layout.addWidget(self.finetune_jsonl_edit)
        btn_layout_dataset = QHBoxLayout()
        btn_browse_jsonl = QPushButton("Select JSONL")
        btn_browse_jsonl.setToolTip("Select JSONL dataset file.")
        btn_browse_jsonl.clicked.connect(self._browse_finetune_jsonl)
        btn_layout_dataset.addWidget(btn_browse_jsonl)
        btn_layout_dataset.addStretch()
        dataset_layout.addLayout(btn_layout_dataset)
        btn_prepare = QPushButton("Prepare dataset (encode)")
        btn_prepare.setToolTip("Encode audio into codec tokens using 12Hz tokenizer.")
        btn_prepare.clicked.connect(self._prepare_finetune_data)
        dataset_layout.addWidget(btn_prepare)
        layout.addWidget(dataset_group)

        train_group = QGroupBox("Training Parameters")
        train_layout = QGridLayout(train_group)
        train_layout.addWidget(QLabel("Speaker name:"), 0, 0)
        self.finetune_speaker_name = QLineEdit("custom_voice")
        self.finetune_speaker_name.setMinimumWidth(150)
        self.finetune_speaker_name.setToolTip("Speaker name for the custom model.")
        train_layout.addWidget(self.finetune_speaker_name, 0, 1)
        train_layout.addWidget(QLabel("Epochs:"), 1, 0)
        self.finetune_epochs = QSpinBox()
        self.finetune_epochs.setRange(1, 100)
        self.finetune_epochs.setValue(3)
        self.finetune_epochs.setMinimumWidth(80)
        self.finetune_epochs.setToolTip("Number of full passes through dataset. Recommended: 3-5.")
        train_layout.addWidget(self.finetune_epochs, 1, 1)
        train_layout.addWidget(QLabel("Batch size:"), 2, 0)
        self.finetune_batch_size = QSpinBox()
        self.finetune_batch_size.setRange(1, 32)
        self.finetune_batch_size.setValue(2)
        self.finetune_batch_size.setMinimumWidth(80)
        self.finetune_batch_size.setToolTip("Batch size. Depends on VRAM: 2-4 on 12GB.")
        train_layout.addWidget(self.finetune_batch_size, 2, 1)
        train_layout.addWidget(QLabel("Learning rate:"), 3, 0)
        self.finetune_lr = QDoubleSpinBox()
        self.finetune_lr.setRange(0.00001, 0.001)
        self.finetune_lr.setValue(0.00002)
        self.finetune_lr.setDecimals(5)
        self.finetune_lr.setPrefix("0.")
        self.finetune_lr.setMinimumWidth(120)
        self.finetune_lr.setToolTip("Learning rate. Recommended: 2e-5.")
        train_layout.addWidget(self.finetune_lr, 3, 1)
        train_layout.addWidget(QLabel("Output path:"), 4, 0)
        self.finetune_output_path = QLineEdit()
        self.finetune_output_path.setPlaceholderText("Path to save the model")
        self.finetune_output_path.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.finetune_output_path.setToolTip("Folder where the fine-tuned model will be saved.")
        train_layout.addWidget(self.finetune_output_path, 4, 1)
        layout.addWidget(train_group)

        self.btn_finetune = QPushButton("Start Fine-Tuning", objectName="finetuneBtn")
        self.btn_finetune.setObjectName("finetuneBtn")
        self.btn_finetune.setFixedHeight(48)
        self.btn_finetune.setMinimumWidth(300)
        self.btn_finetune.setToolTip("Start fine-tuning. Requires GPU >= 12GB VRAM.")
        self.btn_finetune.setStyleSheet("QPushButton#finetuneBtn { background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ef4444, stop:1 #533483); font-size: 15px; font-weight: bold; padding: 10px 24px; }")
        self.btn_finetune.clicked.connect(self._start_finetune)
        layout.addWidget(self.btn_finetune, 0, Qt.AlignmentFlag.AlignCenter)

        self.finetune_progress = QProgressBar()
        self.finetune_progress.setVisible(False)
        layout.addWidget(self.finetune_progress)

        log_group = QGroupBox("Training Log")
        log_layout = QVBoxLayout(log_group)
        self.finetune_log_text = QPlainTextEdit()
        self.finetune_log_text.setReadOnly(True)
        self.finetune_log_text.setMaximumHeight(200)
        self.finetune_log_text.setToolTip("Training log. Shows status, errors, and progress.")
        self.finetune_log_text.setStyleSheet("QPlainTextEdit { background-color: #0a0a1a; color: #c8f7c5; border: 1px solid #3d3d5c; border-radius: 6px; padding: 8px; font-family: Consolas, monospace; font-size: 11px; }")
        log_layout.addWidget(self.finetune_log_text)
        btn_clear_finetune_log = QPushButton("Clear log")
        btn_clear_finetune_log.setToolTip("Clear the training log window.")
        btn_clear_finetune_log.clicked.connect(self.finetune_log_text.clear)
        log_layout.addWidget(btn_clear_finetune_log)
        layout.addWidget(log_group)
        layout.addStretch()
        tab.setLayout(layout)
        self.tabs.addTab(tab, "Fine-tuning")
        self.finetune_thread = None

    def _browse_finetune_model(self):
        path = QFileDialog.getExistingDirectory(self, "Select Model Folder")
        if path: self.finetune_model_path_edit.setText(path)

    def _browse_finetune_jsonl(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select JSONL Dataset", "", "JSONL Files (*.jsonl);;All Files (*)")
        if path: self.finetune_jsonl_edit.setText(path)

    def _prepare_finetune_data(self):
        input_jsonl = self.finetune_jsonl_edit.text().strip()
        if not input_jsonl or not os.path.exists(input_jsonl):
            QMessageBox.warning(self, "Attention", "Specify input JSONL file.")
            return
        default_output = os.path.join(os.path.dirname(input_jsonl), os.path.basename(input_jsonl).replace('.jsonl', '_encoded.jsonl'))
        output_jsonl, _ = QFileDialog.getSaveFileName(self, 'Save prepared dataset', default_output, 'JSONL Files (*.jsonl)')
        if not output_jsonl: return
        model_path = self.finetune_model_path_edit.text().strip()
        if not model_path: tokenizer_path = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
        else:
            tokenizer_path = os.path.join(model_path, "Qwen3-TTS-Tokenizer-12Hz")
            if not os.path.exists(tokenizer_path): tokenizer_path = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
        self.finetune_log_text.appendPlainText(f'[prepare] Input: {input_jsonl}\n[prepare] Output: {output_jsonl}\n[prepare] Tokenizer: {tokenizer_path}\n')
        script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
        prepare_script = os.path.join(script_dir, "finetuning", "prepare_data.py")
        if not os.path.exists(prepare_script):
            QMessageBox.critical(self, "Error", f"prepare_data.py not found:\n{prepare_script}")
            return
        cmd = [sys.executable, prepare_script, '--input_jsonl', input_jsonl, '--output_jsonl', output_jsonl, '--tokenizer_model_path', tokenizer_path, '--device', 'cuda:0']
        self.finetune_log_text.appendPlainText('[prepare] Running...')
        self.finetune_progress.setVisible(True)
        try:
            import subprocess
            process = subprocess.run(cmd, capture_output=True, text=True, cwd=script_dir)
            if process.stdout: self.finetune_log_text.appendPlainText(f'[prepare] {process.stdout.strip()}')
            if process.stderr: self.finetune_log_text.appendPlainText(f'[prepare ERR] {process.stderr.strip()}')
            if process.returncode == 0:
                self.finetune_log_text.appendPlainText('[prepare] OK Done!')
                self.finetune_jsonl_edit.setText(output_jsonl)
                QMessageBox.information(self, "Success", "Dataset prepared!")
            else:
                QMessageBox.critical(self, 'Error', f'Preparation error:\n{process.stderr}')
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Run error:\n{e}')
        finally:
            self.finetune_progress.setVisible(False)

    def _start_finetune(self):
        if not self.finetune_model_path_edit.text().strip():
            QMessageBox.warning(self, "Attention", "Specify model path.")
            return
        if not self.finetune_jsonl_edit.text().strip():
            QMessageBox.warning(self, "Attention", "Specify JSONL dataset path.")
            return
        if not self.finetune_output_path.text().strip():
            QMessageBox.warning(self, "Attention", "Specify output path.")
            return
        self.finetune_log_text.appendPlainText('\n' + '='*50 + '\n')
        self.finetune_log_text.appendPlainText(f'[finetune] Model: {self.finetune_model_path_edit.text()}')
        self.finetune_log_text.appendPlainText(f'[finetune] Dataset: {self.finetune_jsonl_edit.text()}')
        self.finetune_log_text.appendPlainText(f'[finetune] Output: {self.finetune_output_path.text()}')
        self.finetune_log_text.appendPlainText(f'[finetune] Speaker: {self.finetune_speaker_name.text()}')
        self.finetune_log_text.appendPlainText(f'[finetune] Epochs: {self.finetune_epochs.value()}')
        self.finetune_log_text.appendPlainText(f'[finetune] Batch: {self.finetune_batch_size.value()}')
        self.finetune_log_text.appendPlainText(f'[finetune] LR: {self.finetune_lr.value()}')
        self.finetune_log_text.appendPlainText('='*50 + '\n')
        self.finetune_progress.setVisible(True)
        self.btn_finetune.setEnabled(False)
        self.btn_finetune.setText("Training...")
        self._set_status("Fine-tuning...", "#fbbf24")
        script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
        train_script = os.path.join(script_dir, "finetuning", "sft_12hz.py")
        if not os.path.exists(train_script):
            QMessageBox.critical(self, "Error", f"sft_12hz.py not found:\n{train_script}")
            self.finetune_progress.setVisible(False)
            self.btn_finetune.setEnabled(True)
            self.btn_finetune.setText("Start Fine-Tuning")
            return
        cmd = [sys.executable, train_script, '--init_model_path', self.finetune_model_path_edit.text(), '--output_model_path', self.finetune_output_path.text(), '--train_jsonl', self.finetune_jsonl_edit.text(), '--speaker_name', self.finetune_speaker_name.text(), '--num_epochs', str(self.finetune_epochs.value()), '--batch_size', str(self.finetune_batch_size.value()), '--lr', str(self.finetune_lr.value())]
        try:
            import subprocess
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=script_dir, universal_newlines=True)
            def read_output():
                for line in process.stdout:
                    self.finetune_log_text.appendPlainText(f"[sft] {line.strip()}")
                    scrollbar = self.finetune_log_text.verticalScrollBar()
                    scrollbar.setValue(scrollbar.maximum())
                process.wait()
                if process.returncode == 0:
                    self.finetune_log_text.appendPlainText("\n[finetune] OK Training complete!")
                else:
                    self.finetune_log_text.appendPlainText("\n[finetune] ERROR Training failed!")
                self.btn_finetune.setEnabled(True)
                self.btn_finetune.setText("Start Fine-Tuning")
                self.finetune_progress.setVisible(False)
                if process.returncode == 0: self._set_status("Training complete", "#4ade80")
                else: self._set_status("Training failed", "#ef4444")
            finetune_worker = threading.Thread(target=read_output, daemon=True)
            finetune_worker.start()
        except Exception as e:
            self.finetune_log_text.appendPlainText(f"[finetune] Error: {e}")
            QMessageBox.critical(self, "Error", f"Cannot start fine-tuning:\n{e}")
            self.btn_finetune.setEnabled(True)
            self.btn_finetune.setText("Start Fine-Tuning")
            self.finetune_progress.setVisible(False)

