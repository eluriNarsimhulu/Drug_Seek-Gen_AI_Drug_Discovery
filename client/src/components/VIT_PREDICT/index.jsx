import React, { useState, useEffect } from 'react';
import './VIT.css';

function App() {
  const [smiles, setSmiles] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!smiles.trim()) {
      setError('Please enter a SMILES string');
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('http://localhost:5000/vit/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ smiles }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Prediction failed');
      }
      
      const data = await response.json();
      setPrediction(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
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
    <div className='vit'>
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
    <div className="vitApp">
      <div className="vitContainer">
        <header className="vitHeader">
          <h1 className="vitTitle">Molecular Property Classifier</h1>
          <div className="vitSubtitle">Vision Transformer (ViT) Model</div>
        </header>
        
        <main className="vitMainContent">
          <form onSubmit={handleSubmit} className="vitForm">
            <div className="vitFormGroup">
              <label htmlFor="smiles" className="vitLabel">SMILES String:</label>
              <div className="vitInputContainer">
                <input
                  type="text"
                  id="smiles"
                  className="vitInput"
                  value={smiles}
                  onChange={(e) => setSmiles(e.target.value)}
                  placeholder="e.g., CCO for ethanol"
                  required
                />
              </div>
            </div>
            <button type="submit" className={`vitButton ${loading ? 'vitButtonLoading' : ''}`} disabled={loading}>
              {loading ? (
                <>
                  <span className="vitSpinner"></span>
                  <span>Processing...</span>
                </>
              ) : (
                'Predict Properties'
              )}
            </button>
          </form>
          
          {error && (
            <div className="vitErrorMessage">
              <div className="vitErrorIcon">!</div>
              <div className="vitErrorText">
                <strong>Error:</strong> {error}
              </div>
            </div>
          )}
          
          {prediction && !error && (
            <div className="vitResultCard">
              <div className="vitResultHeader">
                <div className="vitResultIcon">✓</div>
                <h2 className="vitResultTitle">Prediction Result</h2>
              </div>
              <div className="vitResultContent">
                <div className="vitResultItem">
                  <span className="vitResultLabel">SMILES:</span>
                  <span className="vitResultValue vitSmiles">{prediction.smiles}</span>
                </div>
                <div className="vitResultItem">
                  <span className="vitResultLabel">Predicted Class:</span>
                  <span className="vitResultValue vitClass">{prediction.predicted_class}</span>
                </div>
              </div>
            </div>
          )}
        </main>
        
        <footer className="vitFooter">
          <div className="vitFooterText">Powered by Vision Transformer Technology</div>
        </footer>
      </div>
    </div>
    </div>
  );
}

export default App;