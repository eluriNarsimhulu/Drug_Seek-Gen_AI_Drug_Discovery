import React, { useState , useEffect} from 'react';
import axios from 'axios';

const Visualization = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);

  const uploadNiiFile = async (file) => {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:8000/api/upload_nii', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        withCredentials: false
      });
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.detail || 'Failed to upload NIfTI file');
    }
  };

  const uploadImage = async (file, modelType) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model_type', modelType);

    try {
      const response = await axios.post('http://localhost:8000/api/upload_image', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        withCredentials: false
      });
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.detail || 'Failed to upload image');
    }
  };

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
    setUploadError(null);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragActive(false);
    const file = e.dataTransfer.files[0];
    setSelectedFile(file);
    setUploadError(null);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragActive(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setDragActive(false);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setUploadError('No file selected.');
      return;
    }

    setUploading(true);
    setUploadError(null);

    try {
      let response;
      if (selectedFile.name.endsWith('.nii.gz')) {
        response = await uploadNiiFile(selectedFile);
      } else {
        response = await uploadImage(selectedFile, 'lung');
      }

      if (response && response.success) {
        // Redirect to viewer with the image paths
        const originalPath = encodeURIComponent(response.original_path || response.mask_path);
        const predictedPath = encodeURIComponent(response.predicted_path || '');
        
        window.location.href = `http://localhost:8000/viewer?original=${originalPath}&predicted=${predictedPath}`;
      } else {
        setUploadError(response?.error || 'Upload failed');
      }
    } catch (error) {
      setUploadError(error.message || 'Upload error');
    } finally {
      setUploading(false);
    }
  };

  const [molecules, setMolecules] = useState([]);

  useEffect(() => {
    const generateMolecules = () => {
      const newMolecules = [];
      for (let i = 0; i < 15; i++) {
        newMolecules.push({
          id: i,
          left: Math.random() * 100,
          top: Math.random() * 100,
          size: Math.random() * 40 + 20,
          duration: Math.random() * 40 + 20,
          delay: Math.random() * 10
        });
      }
      setMolecules(newMolecules);
    };
    
    generateMolecules();
  }, []);


  return (
    <div className="min-h-screen py-12 px-4 ">
      <div className="absolute inset-0 opacity-10">
        {molecules.map(molecule => (
          <div 
            key={molecule.id}
            className="absolute bg-white rounded-full"
            style={{
              left: `${molecule.left}%`,
              top: `${molecule.top}%`,
              width: `${molecule.size}px`,
              height: `${molecule.size}px`,
              animation: `float ${molecule.duration}s ease-in-out ${molecule.delay}s infinite`
            }}
          />
        ))}
      </div>
      <div className="max-w-4xl mx-auto">
        <div className="bg-white bg-opacity-95 backdrop-blur-md rounded-3xl shadow-2xl overflow-hidden">
          {/* Header */}
          <div className="bg-gradient-to-r from-purple-700 to-indigo-800 text-white p-10 relative">
            <div className="absolute inset-0 opacity-10 pattern-dots"></div>
            <h1 className="text-4xl font-extrabold tracking-tight text-center mb-2 relative z-10 drop-shadow-lg">3D Diagnosis Viewer</h1>
            <p className="text-xl opacity-80 text-center relative z-10">Visualize and interact with 3D medical imaging data</p>
          </div>

          {/* Main Content */}
          <div className="p-10">
            {/* Upload Area */}
            <div 
              className={`border-2 border-dashed rounded-xl p-8 text-center mb-8 transition-all ${
                dragActive 
                  ? 'border-indigo-500 bg-indigo-50' 
                  : 'border-gray-300 hover:border-indigo-300 hover:bg-indigo-50/50'
              }`}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
            >
              <input
                type="file"
                accept=".nii.gz,image/*"
                hidden
                id="fileUpload"
                onChange={handleFileChange}
              />
              <label 
                htmlFor="fileUpload" 
                className="flex flex-col items-center justify-center cursor-pointer"
              >
                <div className="w-16 h-16 rounded-full bg-indigo-100 text-indigo-600 flex items-center justify-center mb-4">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                  </svg>
                </div>
                <span className="text-lg font-medium text-gray-700">Drag & drop or Browse files</span>
                <span className="text-sm text-gray-500 mt-2">Support for .nii.gz files and images</span>
              </label>

              {selectedFile && (
                <div className="mt-4 py-3 px-4 bg-indigo-50 rounded-lg inline-block">
                  <p className="text-indigo-800 font-medium">Selected: {selectedFile.name}</p>
                </div>
              )}
            </div>

            {/* Error Message */}
            {uploadError && (
              <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-xl flex items-start gap-3">
                <div className="flex-shrink-0 w-6 h-6 rounded-full bg-red-500 text-white flex items-center justify-center font-bold">!</div>
                <div className="text-red-800">
                  <p className="font-semibold">Error</p>
                  <p>{uploadError}</p>
                </div>
              </div>
            )}

            {/* Upload Button */}
            <button
              className={`w-full h-14 rounded-xl font-medium text-lg shadow-md flex items-center justify-center gap-2 transition-all ${
                uploading || !selectedFile
                  ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                  : 'bg-gradient-to-r from-indigo-600 to-purple-600 text-white hover:from-indigo-700 hover:to-purple-700 hover:shadow-lg transform hover:-translate-y-1'
              }`}
              onClick={handleUpload}
              disabled={uploading || !selectedFile}
            >
              {uploading ? (
                <>
                  <span className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></span>
                  <span>Processing...</span>
                </>
              ) : (
                selectedFile?.name.endsWith('.nii.gz') ? 'Predict & Visualize' : 'Upload Image'
              )}
            </button>

            {uploading && (
              <p className="mt-4 text-center text-indigo-700">Processing and redirecting...</p>
            )}
          </div>

          {/* Footer */}
          <div className="py-4 px-6 text-center border-t border-gray-200">
            <p className="text-gray-500 text-sm">Powered by Advanced 3D Visualization Technology</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Visualization;