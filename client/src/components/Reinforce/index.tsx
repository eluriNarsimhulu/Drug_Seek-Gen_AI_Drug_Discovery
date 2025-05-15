import React, { useState } from 'react';
import { Beaker, Github, Search, AlertCircle, Info, ArrowUpDown, Download, ExternalLink, Image } from 'lucide-react';

interface Molecule {
  smiles: string;
  pic50?: number;
  logp?: number;
  image?: string | null;
}

interface PredictionResult {
  input_smiles: string;
  input_image?: string | null;
  results: Molecule[];
  type: 'pic50' | 'logp';
}

const Reinforce: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null);
  const [smiles, setSmiles] = useState('');
  const [inputError, setInputError] = useState('');
  const [showInfo, setShowInfo] = useState(false);
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');
  const [showStructures, setShowStructures] = useState(true);

  const handlePrediction = async (inputSmiles: string, type: 'pic50' | 'logp') => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('http://localhost:5000/api/reinforce/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ smiles: inputSmiles, type }),
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Failed to get prediction');

      setPredictionResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unknown error occurred');
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e: React.FormEvent, type: 'pic50' | 'logp') => {
    e.preventDefault();
    if (!smiles.trim()) {
      setInputError('Please enter a SMILES string');
      return;
    }
    setInputError('');
    handlePrediction(smiles, type);
  };

  const toggleSortDirection = () => {
    setSortDirection(prev => (prev === 'desc' ? 'asc' : 'desc'));
  };

  const exportResults = () => {
    if (!predictionResult) return;
    const headers = ['SMILES', predictionResult.type.toUpperCase()];
    const csv = [
      headers,
      ...predictionResult.results.map(m => [
        m.smiles,
        predictionResult.type === 'pic50' ? m.pic50 : m.logp,
      ]),
    ]
      .map(row => row.join(','))
      .join('\n');

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${predictionResult.type}_predictions.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  const exampleSmiles = [
    "CC(=O)Oc1ccccc1C(=O)O", // Aspirin
    "COc1cc2c(cc1OC)C(=NCC2)C(=O)O", // Methyldopa
    "CC(C)NCC(O)COc1cccc2ccccc12", // Propranolol
  ];

  const sortedResults = predictionResult
    ? [...predictionResult.results].sort((a, b) => {
        const aVal = predictionResult.type === 'pic50' ? a.pic50 || 0 : a.logp || 0;
        const bVal = predictionResult.type === 'pic50' ? b.pic50 || 0 : b.logp || 0;
        return sortDirection === 'desc' ? bVal - aVal : aVal - bVal;
      })
    : [];

  const getPubChemUrl = (smiles: string) => `https://pubchem.ncbi.nlm.nih.gov/compound/${encodeURIComponent(smiles)}`;

  return (
    <div className="min-h-screen ">
      
      <div className="reinforce-app">
        <div className="reinforce-container">
          {/* Header */}
          <header className="reinforce-header">
            <h1 className="reinforce-title">Molecular Property Predictor</h1>
            <div className="reinforce-subtitle">Vision-Guided Molecular Analysis</div>
          </header>
          
          <main className="reinforce-main">
            {/* Input Form */}
            <div className="reinforce-card mb-6">
              <div className="flex justify-between items-center mb-4">
                <div className="flex items-center">
                  <Beaker className="w-6 h-6 mr-2 text-purple-700" />
                  <h2 className="text-xl font-bold text-purple-800">Molecule Input</h2>
                </div>
                <button 
                  onClick={() => setShowInfo(!showInfo)} 
                  className="reinforce-info-button"
                >
                  <Info className="w-5 h-5" />
                </button>
              </div>
              
              {showInfo && (
                <div className="reinforce-info-box">
                  <div className="reinforce-info-icon">ℹ</div>
                  <div className="reinforce-info-content">
                    <h3 className="font-semibold mb-1">About SMILES Notation</h3>
                    <p className="mb-2">
                      SMILES (Simplified Molecular Input Line Entry System) is a way to represent chemical structures
                      using ASCII strings.
                    </p>
                    <p>Examples: <code>CC(=O)O</code>, <code>c1ccccc1</code></p>
                  </div>
                </div>
              )}
              
              <form onSubmit={(e) => handleSubmit(e, 'pic50')}>
                <div className="reinforce-form-group">
                  <label htmlFor="smiles" className="reinforce-label">SMILES String:</label>
                  <div className="reinforce-input-container">
                    <Search className="absolute left-3 top-3 text-gray-400 w-5 h-5" />
                    <input
                      type="text"
                      id="smiles"
                      className={`reinforce-input ${inputError ? 'border-red-300' : ''}`}
                      value={smiles}
                      onChange={(e) => setSmiles(e.target.value)}
                      placeholder="e.g., CCO for ethanol"
                      disabled={loading}
                    />
                  </div>
                  
                  {inputError && (
                    <div className="flex items-center mt-2 text-red-600 text-sm">
                      <AlertCircle className="w-4 h-4 mr-1" />
                      <span>{inputError}</span>
                    </div>
                  )}
                </div>
                
                <div className="reinforce-button-group">
                  <button
                    type="submit"
                    disabled={loading}
                    className="reinforce-button reinforce-button-primary"
                  >
                    {loading ? (
                      <>
                        <span className="reinforce-spinner"></span>
                        <span>Processing...</span>
                      </>
                    ) : (
                      'Predict pIC50'
                    )}
                  </button>
                  
                  <button
                    type="button"
                    onClick={(e) => handleSubmit(e, 'logp')}
                    disabled={loading}
                    className="reinforce-button reinforce-button-secondary"
                  >
                    {loading ? 'Processing...' : 'Predict LogP'}
                  </button>
                </div>
              </form>
              
              <div className="mt-6">
                <h3 className="text-sm font-medium text-gray-600 mb-2">Example SMILES:</h3>
                <div className="flex flex-wrap gap-2">
                  {exampleSmiles.map((ex, i) => (
                    <button
                      key={i}
                      onClick={() => setSmiles(ex)}
                      disabled={loading}
                      className="reinforce-example-button"
                    >
                      {ex}
                    </button>
                  ))}
                </div>
              </div>
            </div>
            
            {/* Error Message */}
            {error && (
              <div className="reinforce-error-message">
                <div className="reinforce-error-icon">!</div>
                <div className="reinforce-error-text">
                  <strong>Error:</strong> {error}
                </div>
              </div>
            )}
            
            {/* Loading State */}
            {loading && (
              <div className="reinforce-card flex justify-center items-center py-12">
                <div className="flex flex-col items-center">
                  <div className="w-12 h-12 border-4 border-purple-200 border-t-purple-600 rounded-full animate-spin mb-4"></div>
                  <p className="text-purple-700">Generating predictions...</p>
                </div>
              </div>
            )}
            
            {/* Results Section */}
            {predictionResult && !loading && (
              <div className="space-y-6">
                {/* Input Molecule Structure */}
                {predictionResult.input_image && (
                  <div className="reinforce-card">
                    <h2 className="reinforce-section-title mb-4">Input Molecule Structure</h2>
                    <div className="flex justify-center">
                      <div className="border border-gray-200 rounded-md p-4 bg-white">
                        <img 
                          src={predictionResult.input_image} 
                          alt="Input molecule structure" 
                          className="max-h-64 w-auto"
                        />
                      </div>
                    </div>
                  </div>
                )}
                
                {/* Results Table */}
                <div className="reinforce-result-card">
                  <div className="reinforce-result-header">
                    <div className="reinforce-result-icon">✓</div>
                    <h2 className="reinforce-result-title">
                      {predictionResult.type === 'pic50' ? 'pIC50 Prediction Results' : 'LogP Prediction Results'}
                    </h2>
                    <div className="flex items-center ml-auto space-x-4">
                      <button 
                        onClick={() => setShowStructures(!showStructures)} 
                        className="reinforce-action-button"
                      >
                        <Image className="w-4 h-4 mr-1" />
                        {showStructures ? 'Hide Structures' : 'Show Structures'}
                      </button>
                      <button 
                        onClick={exportResults} 
                        className="reinforce-action-button"
                      >
                        <Download className="w-4 h-4 mr-1" />
                        Export CSV
                      </button>
                    </div>
                  </div>

                  <div className="overflow-x-auto">
                    <table className="reinforce-table">
                      <thead>
                        <tr>
                          <th className="w-12">#</th>
                          {showStructures && (
                            <th className="w-32">Structure</th>
                          )}
                          <th>SMILES</th>
                          <th onClick={toggleSortDirection} className="cursor-pointer w-32">
                            <div className="flex items-center justify-center">
                              {predictionResult.type.toUpperCase()} 
                              <ArrowUpDown className="ml-1 w-4 h-4" />
                            </div>
                          </th>
                          <th className="w-24">Lookup</th>
                        </tr>
                      </thead>
                      <tbody>
                        {sortedResults.map((molecule, index) => (
                          <tr key={index}>
                            <td className="text-center">{index + 1}</td>
                            {showStructures && (
                              <td>
                                {molecule.image ? (
                                  <div className="flex justify-center">
                                    <img 
                                      src={molecule.image} 
                                      alt={`Structure of ${molecule.smiles}`} 
                                      className="h-20 w-auto"
                                    />
                                  </div>
                                ) : (
                                  <div className="text-center text-gray-400 text-sm">
                                    No structure
                                  </div>
                                )}
                              </td>
                            )}
                            <td className="font-mono text-sm break-all">{molecule.smiles}</td>
                            <td className="text-center">
                              {(predictionResult.type === 'pic50' ? molecule.pic50 : molecule.logp)?.toFixed(3)}
                            </td>
                            <td>
                              <a
                                href={getPubChemUrl(molecule.smiles)}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="reinforce-link flex items-center justify-center"
                              >
                                PubChem <ExternalLink className="ml-1 w-3 h-3" />
                              </a>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}
            
            {/* About Box */}
            <div className="reinforce-about-box">
              <div className="reinforce-about-icon">
                <Beaker className="w-6 h-6" />
              </div>
              <h2 className="reinforce-about-title">About This Tool</h2>
              <div className="reinforce-about-content">
                <p className="mb-3">
                  This application predicts molecular properties using pre-trained models:
                </p>
                <ul className="list-disc pl-5 space-y-1 mb-4">
                  <li>pIC50: Predicts activity values and generates similar molecules</li>
                  <li>LogP: Predicts the partition coefficient, a measure of lipophilicity</li>
                </ul>
                <h3 className="font-semibold mb-2">How to use:</h3>
                <ol className="list-decimal pl-5 space-y-1">
                  <li>Enter a valid SMILES string</li>
                  <li>Select pIC50 or LogP</li>
                  <li>View results, structures and properties</li>
                  <li>Export or lookup in PubChem</li>
                </ol>
              </div>
            </div>
          </main>
          
          <footer className="reinforce-footer">
            <div className="reinforce-footer-text">
              Powered by Vision-Guided Molecular Analysis Technology
            </div>
            <div className="text-xs mt-1 opacity-70">
              © 2025 Molecular Property Predictor - Built for demonstration purposes
            </div>
          </footer>
        </div>
      </div>
      
      {/* Custom CSS - Using Tailwind classes for styling elements */}
      <style jsx>{`
        .reinforce-app {
          width: 100%;
          display: flex;
          justify-content: center;
          padding: 2rem 1rem;
          min-height: 100vh;
        }
        
        .reinforce-container {
          width: 100%;
          max-width: 1024px;
          background: rgba(255, 255, 255, 0.95);
          backdrop-filter: blur(10px);
          border-radius: 24px;
          box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
          overflow: hidden;
        }
        
        .reinforce-header {
          background: linear-gradient(135deg, #8e2de2, #4a00e0);
          color: white;
          padding: 2.5rem 3rem;
          text-align: center;
          position: relative;
          border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .reinforce-title {
          font-size: 2.8rem;
          font-weight: 800;
          margin: 0;
          letter-spacing: -0.025em;
          text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }
        
        .reinforce-subtitle {
          font-size: 1.2rem;
          font-weight: 400;
          opacity: 0.8;
          margin-top: 0.5rem;
        }
        
        .reinforce-main {
          padding: 2.5rem;
        }
        
        .reinforce-card {
          background: white;
          border-radius: 16px;
          box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
          padding: 1.5rem;
          margin-bottom: 1.5rem;
        }
        
        .reinforce-label {
          font-weight: 600;
          margin-bottom: 0.75rem;
          color: #4a00e0;
          font-size: 1.05rem;
          display: block;
        }
        
        .reinforce-input-container {
          position: relative;
        }
        
        .reinforce-input {
          width: 100%;
          padding: 0.75rem 1.25rem 0.75rem 2.5rem;
          border: 2px solid #e5e7eb;
          border-radius: 12px;
          font-size: 1.05rem;
          box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
          transition: all 0.25s ease;
        }
        
        .reinforce-input:focus {
          outline: none;
          border-color: #8e2de2;
          box-shadow: 0 0 0 3px rgba(142, 45, 226, 0.15);
        }
        
        .reinforce-button-group {
          display: flex;
          gap: 1rem;
          margin-top: 1.5rem;
        }
        
        .reinforce-button {
          display: flex;
          justify-content: center;
          align-items: center;
          gap: 0.5rem;
          border: none;
          padding: 0.75rem 1.5rem;
          border-radius: 12px;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.25s ease;
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }
        
        .reinforce-button-primary {
          background: linear-gradient(135deg, #8e2de2, #4a00e0);
          color: white;
        }
        
        .reinforce-button-secondary {
          background: linear-gradient(135deg, #667eea, #764ba2);
          color: white;
        }
        
        .reinforce-button:hover {
          transform: translateY(-2px);
          box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        }
        
        .reinforce-spinner {
          width: 20px;
          height: 20px;
          border: 3px solid rgba(255, 255, 255, 0.3);
          border-radius: 50%;
          border-top-color: white;
          animation: spin 1s ease-in-out infinite;
        }
        
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
        
        .reinforce-example-button {
          text-align: center;
          padding: 0.375rem 0.75rem;
          background-color: #f3f4f6;
          border-radius: 8px;
          font-size: 0.75rem;
          transition: all 0.2s;
        }
        
        .reinforce-example-button:hover {
          background-color: #e5e7eb;
        }
        
        .reinforce-info-button {
          color: #8e2de2;
          transition: color 0.2s;
        }
        
        .reinforce-info-button:hover {
          color: #4a00e0;
        }
        
        .reinforce-info-box {
          margin: 1rem 0;
          padding: 1rem;
          background-color: #f5f3ff;
          border-radius: 12px;
          display: flex;
          gap: 0.75rem;
          border: 1px solid #ddd6fe;
        }
        
        .reinforce-info-icon {
          background-color: #8e2de2;
          color: white;
          width: 24px;
          height: 24px;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          font-weight: bold;
          flex-shrink: 0;
        }
        
        .reinforce-error-message {
          margin: 1.5rem 0;
          padding: 1.25rem;
          background-color: #fff5f5;
          color: #e53e3e;
          border-radius: 12px;
          display: flex;
          gap: 1rem;
          align-items: flex-start;
          border: 1px solid #fed7d7;
          box-shadow: 0 4px 10px rgba(229, 62, 62, 0.1);
        }
        
        .reinforce-error-icon {
          background-color: #fc8181;
          color: white;
          width: 24px;
          height: 24px;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          font-weight: bold;
          flex-shrink: 0;
        }
        
        .reinforce-section-title {
          font-size: 1.25rem;
          font-weight: 700;
          color: #4a5568;
        }
        
        .reinforce-result-card {
          background: white;
          border-radius: 16px;
          overflow: hidden;
          box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05), 0 0 0 1px rgba(0, 0, 0, 0.05);
        }
        
        .reinforce-result-header {
          background: linear-gradient(135deg, #8e2de2, #4a00e0);
          color: white;
          padding: 1rem 1.5rem;
          display: flex;
          align-items: center;
          gap: 0.75rem;
        }
        
        .reinforce-result-icon {
          background-color: rgba(255, 255, 255, 0.3);
          border-radius: 50%;
          width: 28px;
          height: 28px;
          display: flex;
          align-items: center;
          justify-content: center;
          font-weight: bold;
        }
        
        .reinforce-result-title {
          margin: 0;
          font-size: 1.25rem;
          font-weight: 600;
        }
        
        .reinforce-action-button {
          display: flex;
          align-items: center;
          color: rgba(255, 255, 255, 0.9);
          font-size: 0.875rem;
          transition: color 0.2s;
        }
        
        .reinforce-action-button:hover {
          color: white;
        }
        
        .reinforce-table {
          width: 100%;
          border-collapse: collapse;
        }
        
        .reinforce-table th {
          background-color: #f7fafc;
          padding: 0.75rem 1rem;
          font-size: 0.75rem;
          font-weight: 600;
          color: #4a5568;
          text-transform: uppercase;
          letter-spacing: 0.05em;
          border-bottom: 1px solid #e2e8f0;
        }
        
        .reinforce-table td {
          padding: 1rem;
          border-bottom: 1px solid #e2e8f0;
          font-size: 0.875rem;
          color: #4a5568;
        }
        
        .reinforce-table tr:last-child td {
          border-bottom: none;
        }
        
        .reinforce-link {
          color: #8e2de2;
          font-size: 0.875rem;
          transition: color 0.2s;
        }
        
        .reinforce-link:hover {
          color: #4a00e0;
          text-decoration: underline;
        }
        
        .reinforce-about-box {
          background: #f8f7ff;
          border-radius: 16px;
          padding: 1.5rem;
          border: 1px solid #e9e3ff;
          margin-top: 2rem;
        }
        
        .reinforce-about-icon {
          background-color: #8e2de2;
          color: white;
          width: 40px;
          height: 40px;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          margin-bottom: 1rem;
        }
        
        .reinforce-about-title {
          font-size: 1.25rem;
          font-weight: 700;
          color: #4a5568;
          margin-bottom: 1rem;
        }
        
        .reinforce-about-content {
          font-size: 0.875rem;
          color: #4a5568;
        }
        
        .reinforce-footer {
          padding: 1.5rem;
          text-align: center;
          color: #718096;
          border-top: 1px solid #edf2f7;
          background-color: #f7fafc;
        }
        
        .reinforce-footer-text {
          opacity: 0.8;
          font-size: 0.9rem;
        }
        
        @media (max-width: 768px) {
          .reinforce-header {
            padding: 2rem 1.5rem;
          }
          
          .reinforce-title {
            font-size: 2rem;
          }
          
          .reinforce-main {
            padding: 1.5rem;
          }
          
          .reinforce-button-group {
            flex-direction: column;
          }
        }
      `}</style>
    </div>
  );
};

export default Reinforce;