## 🚀 Step-by-Step Hugging Face Upload Instructions

### ✅ Pre-Upload Checklist (COMPLETED)
- [x] Dataset structure validated (626 audio files, 8236 annotations)
- [x] Required packages installed
- [x] HuggingFace Hub ready

### 🔐 Step 1: Get Your Hugging Face Token

1. Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click "New token"
3. Choose "Write" access (required for uploading datasets)
4. Give it a name like "Dataset Upload Token"
5. Copy the token (keep it safe!)

### 📤 Step 2: Run the Upload Script

```bash
python upload_to_huggingface.py
```

### 🎯 Step 3: Follow the Interactive Prompts

The script will ask you for:

1. **🤔 Proceed with upload?** 
   - Type `y` or `yes`

2. **📝 Custom dataset name?**
   - Press Enter for default: `synthetic-multilingual-speaker-diarization`
   - Or type your preferred name (e.g., `my-hindi-english-punjabi-dataset`)

3. **🔐 HuggingFace Token**
   - Paste your token from Step 1
   - Token will be saved for future uploads

### 📊 What Happens During Upload

```
🔄 Loading dataset from: ./output_diarization_dataset
📊 Loaded 8236 annotations from CSV
🎵 Found 626 unique audio files
🔄 Processing audio files...
   Processed 50/626 files...
   Processed 100/626 files...
   ...
✅ Successfully processed 626 audio files
📊 Creating dataset splits...
📝 Creating dataset card...
🏗️ Creating repository...
📤 Uploading dataset...
```

### 🎉 After Successful Upload

Your dataset will be available at:
```
https://huggingface.co/datasets/{your-username}/{dataset-name}
```

### 🔧 If You Get Errors

**Common fixes:**

1. **Network timeout**: Re-run the script (it resumes from where it left off)
2. **Token error**: Make sure token has "Write" permissions
3. **Memory issues**: Close other programs to free up RAM

### 📱 Quick Commands

```bash
# Test everything first (optional)
python test_upload.py

# Run the actual upload
python upload_to_huggingface.py
```

### 💡 Tips

- The upload may take 30-60 minutes for 626 files
- You can cancel anytime with Ctrl+C
- The script shows progress as it uploads
- Your dataset will be public by default (great for sharing!)

Ready to upload? Just run: `python upload_to_huggingface.py` 🚀