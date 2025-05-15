import React, { useState, useEffect } from 'react';
import { Beaker, Search, ArrowRight, Loader, AlertCircle, CheckCircle2, ChevronDown } from 'lucide-react';

const Chembert = () => {
  const [userInput, setUserInput] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [submittedInput, setSubmittedInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showExamples, setShowExamples] = useState(false);
  const [showInfo, setShowInfo] = useState(true);

  const examples = [
    "CCO<mask>",
    "C1=CC=C(C=C1)C<mask>",
    "CC(C)=CCCC(C)=CC<mask>"
  ];

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!userInput.trim()) {
      setError('Please enter a SMILES string with a <mask> tag');
      return;
    }
    
    if (!userInput.includes('<mask>')) {
      setError('Your input must include a <mask> tag');
      return;
    }
    
    setLoading(true);
    setPrediction(null);
    setError(null);

    try {
      const response = await fetch('http://localhost:5000/chemberta/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ user_input: userInput }),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || 'Failed to get prediction');
      }

      // Transform prediction data to array format if it's not already
      let formattedPrediction;
      if (typeof data.prediction === 'string') {
        // Extract predictions from string (assuming format: "2 (0.4612) 1 (0.2972)")
        const regex = /([^ ]+) \(([0-9.]+)\)/g;
        const matches = [...data.prediction.matchAll(regex)];
        formattedPrediction = matches.map(match => ({
          atom: match[1],
          probability: parseFloat(match[2])
        }));
      } else if (Array.isArray(data.prediction)) {
        formattedPrediction = data.prediction;
      } else {
        formattedPrediction = [{ atom: "Error parsing predictions", probability: 0 }];
      }
      
      setPrediction(formattedPrediction);
      setSubmittedInput(userInput);
    } catch (error) {
      console.error('Prediction error:', error);
      setError(error.message || 'An error occurred while predicting.');
    } finally {
      setLoading(false);
    }
  };

  const selectExample = (example) => {
    setUserInput(example);
    setShowExamples(false);
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
    <div className="bg-gradient-to-br  to-indigo-900 min-h-screen py-12 relative overflow-hidden">
      {/* Background molecules for decoration */}
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
      
      <div className="max-w-6xl mx-auto px-4">
        <div className="bg-white bg-opacity-95 backdrop-blur-lg rounded-3xl shadow-2xl overflow-hidden">
          {/* Header */}
          <div className="bg-gradient-to-r from-purple-600 to-indigo-700 text-white px-8 py-10 relative">
            <div className="absolute inset-0 opacity-10 bg-pattern"></div>
            <div className="text-center relative z-10">
              <h1 className="text-4xl font-extrabold tracking-tight mb-2">Masked SMILES Predictor</h1>
              <p className="text-purple-100 text-lg font-medium">ChemBERTa Model for Molecular Prediction</p>
            </div>
          </div>
          
          {/* Main Content */}
          <main className="p-8">
            {/* Info Panel */}
            {showInfo && (
              <div className="bg-indigo-50 border-l-4 border-indigo-500 p-4 mb-8 rounded-md flex items-start">
                <AlertCircle className="h-5 w-5 text-indigo-600 mt-0.5 flex-shrink-0" />
                <div className="ml-3 flex-grow">
                  <p className="text-indigo-700">
                    Use <code className="bg-indigo-100 px-1 py-0.5 rounded text-indigo-800">&lt;mask&gt;</code> to indicate the position you want the model to predict. 
                    For example, <code className="bg-indigo-100 px-1 py-0.5 rounded text-indigo-800">CCO&lt;mask&gt;</code> will predict what might follow ethanol.
                  </p>
                </div>
                <button 
                  onClick={() => setShowInfo(false)}
                  className="text-indigo-500 hover:text-indigo-700 ml-2 flex-shrink-0"
                >
                  ×
                </button>
              </div>
            )}
            
            {/* Input Form */}
            <form onSubmit={handleSubmit} className="mb-8">
              <div className="mb-4 relative">
                <label htmlFor="smiles" className="block font-semibold text-gray-700 mb-2">SMILES String with &lt;mask&gt; tag:</label>
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <Search className="h-5 w-5 text-gray-400" />
                  </div>
                  <input
                    id="smiles"
                    type="text"
                    value={userInput}
                    onChange={(e) => setUserInput(e.target.value)}
                    placeholder="e.g., CCO<mask>"
                    className="block w-full pl-10 pr-4 py-3 border-2 border-gray-300 rounded-xl focus:ring-2 focus:ring-purple-500 focus:border-purple-500 shadow-sm font-mono"
                  />
                </div>
              </div>
              
              <div className="flex flex-col sm:flex-row gap-4">
                <button 
                  type="submit" 
                  disabled={loading}
                  className="flex items-center justify-center py-3 px-6 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 text-white font-medium rounded-xl shadow-md transition-all duration-200 disabled:opacity-70"
                >
                  {loading ? (
                    <>
                      <Loader className="animate-spin h-5 w-5 mr-2" />
                      <span>Processing...</span>
                    </>
                  ) : (
                    <>
                      <ArrowRight className="h-5 w-5 mr-2" />
                      <span>Predict Properties</span>
                    </>
                  )}
                </button>
                
                <div className="relative">
                  <button
                    type="button"
                    onClick={() => setShowExamples(!showExamples)}
                    className="flex items-center justify-center py-3 px-6 bg-gray-100 hover:bg-gray-200 text-gray-700 font-medium rounded-xl transition-all duration-200"
                  >
                    Example Inputs <ChevronDown className={`h-4 w-4 ml-2 transition-transform ${showExamples ? 'rotate-180' : ''}`} />
                  </button>
                  
                  {showExamples && (
                    <div className="absolute z-10 mt-2 w-full bg-white shadow-lg rounded-xl border border-gray-200 overflow-hidden">
                      {examples.map((example, index) => (
                        <button
                          key={index}
                          type="button"
                          onClick={() => selectExample(example)}
                          className="block w-full px-4 py-3 text-left hover:bg-gray-50 font-mono transition-colors border-b border-gray-100 last:border-0"
                        >
                          {example}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </form>
          
            {/* Error message */}
            {error && (
              <div className="bg-red-50 border-l-4 border-red-500 p-4 mb-8 rounded-md">
                <div className="flex">
                  <AlertCircle className="h-5 w-5 text-red-600 flex-shrink-0" />
                  <div className="ml-3">
                    <p className="text-red-700">{error}</p>
                  </div>
                </div>
              </div>
            )}

            {/* Results */}
            {submittedInput && (
              <div className="bg-gray-50 rounded-xl p-6 mb-6">
                <div className="flex items-center mb-3">
                  <CheckCircle2 className="h-6 w-6 text-green-500 mr-2" />
                  <h2 className="text-xl font-semibold text-gray-800">Input SMILES</h2>
                </div>
                <div className="bg-white p-4 rounded-lg border border-gray-200 font-mono text-gray-700">
                  {submittedInput}
                </div>
              </div>
            )}

            {prediction && (
              <div className="bg-gray-50 rounded-xl p-6">
                <div className="flex items-center mb-3">
                  <Beaker className="h-6 w-6 text-purple-500 mr-2" />
                  <h2 className="text-xl font-semibold text-gray-800">Top Predictions</h2>
                </div>
                
                <div className="space-y-3">
                  {Array.isArray(prediction) ? (
                    prediction.slice(0, 5).map((item, index) => (
                      <div 
                        key={index} 
                        className={`p-4 rounded-lg border flex items-center ${index === 0 ? 'bg-gradient-to-r from-purple-50 to-indigo-50 border-purple-200' : 'bg-white border-gray-200'}`}
                      >
                        <div className={`w-8 h-8 rounded-full flex items-center justify-center mr-4 ${index === 0 ? 'bg-purple-500 text-white' : 'bg-gray-200 text-gray-700'} font-semibold`}>
                          {index + 1}
                        </div>
                        <div className="flex-grow">
                          <div className="text-lg font-semibold font-mono">
                            {item.atom}
                          </div>
                          <div className="text-gray-500">
                            {(item.probability * 100).toFixed(2)}% probability
                          </div>
                        </div>
                        <div className="w-24 bg-gray-200 rounded-full h-2 overflow-hidden">
                          <div 
                            className="bg-gradient-to-r from-purple-500 to-indigo-500 h-full" 
                            style={{ width: `${item.probability * 100}%` }}
                          ></div>
                        </div>
                      </div>
                    ))
                  ) : (
                    <div className="p-4 bg-white rounded-lg border border-gray-200">
                      <p className="text-gray-700">Unable to parse prediction data.</p>
                    </div>
                  )}
                </div>
                
                <div className="mt-5 p-4 bg-indigo-50 rounded-md">
                  <p className="text-sm text-indigo-700">
                    The predictions are based on the chemical context and likelihood of specific atoms or groups following the given SMILES pattern.
                  </p>
                </div>
              </div>
            )}
          </main>
          
          {/* Footer */}
          <footer className="bg-gray-50 py-4 px-8 text-center text-gray-500 text-sm border-t border-gray-200">
            Powered by ChemBERTa Transformer Technology
          </footer>
        </div>
      </div>
      
      {/* CSS for animations */}
      <style jsx>{`
        @keyframes float {
          0%, 100% { transform: translateY(0) rotate(0deg); }
          50% { transform: translateY(-20px) rotate(10deg); }
        }
        
        .bg-pattern {
          background-image: url("data:image/svg+xml,%3Csvg width='100' height='100' viewBox='0 0 100 100' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M11 18c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm48 25c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm-43-7c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm63 31c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM34 90c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm56-76c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM12 86c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm28-65c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm23-11c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-6 60c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm29 22c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zM32 63c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm57-13c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5z' fill='%23ffffff' fill-opacity='0.15' fill-rule='evenodd'/%3E%3C/svg%3E");
        }
      `}</style>
    </div>
  );
};

export default Chembert;