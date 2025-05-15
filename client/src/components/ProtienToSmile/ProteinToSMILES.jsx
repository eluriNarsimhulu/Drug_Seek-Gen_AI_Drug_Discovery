import { useState,useEffect } from 'react';
import axios from 'axios';

export default function ProteinToSMILES() {
  const [proteinSequence, setProteinSequence] = useState('');
  const [smilesResult, setSmilesResult] = useState('');
  const [moleculeImageBase64, setMoleculeImageBase64] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError('');
    setSmilesResult('');
    setMoleculeImageBase64('');

    try {
      const response = await axios.post('http://127.0.0.1:5002/predict', {
        protein_sequence: proteinSequence,
      }, {
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (response.data && response.data.smiles) {
        setSmilesResult(response.data.smiles);
        setMoleculeImageBase64(response.data.image);
      } else if (response.data.error) {
        setError(response.data.error);
      } else {
        setError('Invalid response from server');
      }
    } catch (err) {
      console.error(err);
      if (err.response && err.response.data && err.response.data.error) {
        setError(err.response.data.error);
      } else {
        setError('Error connecting to server');
      }
    } finally {
      setLoading(false);
    }
  };

  const exampleSequence = 'MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNGGHFLRILPDGTVDGTRDRSDQH' +
    'IQLQLSAESVGEVYIKSTETGQYLAMDTSGLLYGSQTPSEECLFLERLEENHYN' +
    'TYTSKKHAEKNWFVGLKKNGSCKRGPRTHYGQKAILFLPLPV';
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
    <div className="min-h-screen  py-12 px-4">
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
      <div className="w-full max-w-4xl mx-auto">
        <div className="bg-white bg-opacity-95 backdrop-blur-md rounded-3xl shadow-2xl overflow-hidden">
          
          {/* Header Section */}
          <header className="bg-gradient-to-r from-teal-600 to-emerald-700 text-white py-10 px-12 relative border-b border-white border-opacity-10">
            <div className="absolute inset-0 opacity-5 pattern-dots"></div>
            <h1 className="text-4xl font-extrabold tracking-tight z-10 relative text-shadow">
              Protein to SMILES Converter
            </h1>
            <div className="text-lg opacity-80 mt-2 z-10 relative">
              Powered by Advanced Molecular Prediction
            </div>
          </header>
          
          {/* Main Content */}
          <main className="p-10">
            <form onSubmit={handleSubmit} className="flex flex-col gap-6">
              <div className="flex flex-col">
                <label htmlFor="protein-sequence" className="font-semibold text-teal-800 mb-3 text-lg">
                  Protein Sequence:
                </label>
                <div className="relative">
                  <textarea
                    id="protein-sequence"
                    value={proteinSequence}
                    onChange={(e) => setProteinSequence(e.target.value)}
                    placeholder="Enter protein sequence..."
                    rows="5"
                    required
                    className="w-full p-4 border-2 border-gray-200 rounded-xl font-mono text-base shadow-sm focus:outline-none focus:border-teal-600 focus:ring-2 focus:ring-teal-600 focus:ring-opacity-20 transition-all duration-200"
                  />
                </div>
                <div className="mt-2 text-sm text-gray-600">
                  <span className="font-medium text-teal-700">Example:</span>
                  <button 
                    type="button" 
                    onClick={() => setProteinSequence(exampleSequence)}
                    className="ml-2 text-sm text-purple-600 hover:text-purple-800 underline"
                  >
                    Use example sequence
                  </button>
                </div>
              </div>
              
              <button 
                type="submit" 
                disabled={loading}
                className={`flex justify-center items-center gap-2 bg-gradient-to-r from-teal-500 to-emerald-600 text-white border-none py-4 px-6 rounded-xl text-lg font-semibold cursor-pointer transition-all duration-300 shadow-lg hover:shadow-xl hover:translate-y-px disabled:opacity-70 disabled:cursor-not-allowed h-14 ${loading ? 'bg-gradient-to-r from-teal-600 to-emerald-700' : ''}`}
              >
                {loading ? (
                  <>
                    <span className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></span>
                    <span>Processing...</span>
                  </>
                ) : (
                  'Convert to SMILES'
                )}
              </button>
            </form>
            
            {error && (
              <div className="mt-6 p-5 bg-red-50 text-red-700 rounded-xl flex gap-4 items-start border border-red-200 shadow-sm">
                <div className="bg-red-500 text-white w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0">
                  !
                </div>
                <div>
                  <strong className="font-semibold">Error:</strong> {error}
                </div>
              </div>
            )}
            
            {smilesResult && (
              <div className="mt-10 rounded-2xl overflow-hidden bg-white shadow-lg border border-gray-100 animate-fadeIn">
                <div className="bg-gradient-to-r from-purple-500 to-violet-600 text-white p-5 flex items-center gap-3">
                  <div className="bg-white bg-opacity-30 rounded-full w-8 h-8 flex items-center justify-center font-bold text-xl">
                    ✓
                  </div>
                  <h2 className="text-xl font-semibold m-0">Prediction Result</h2>
                </div>
                
                <div className="p-6">
                  <div className="mb-5">
                    <h3 className="uppercase text-sm font-semibold text-gray-600 mb-2 tracking-wider">
                      Predicted SMILES String:
                    </h3>
                    <div className="p-3 bg-gray-50 rounded-lg font-mono text-base break-all border border-gray-200 text-teal-700">
                      {smilesResult}
                    </div>
                  </div>
                  
                  {moleculeImageBase64 && (
                    <div>
                      <h3 className="uppercase text-sm font-semibold text-gray-600 mb-2 tracking-wider">
                        Generated Molecule:
                      </h3>
                      <div className="bg-gray-50 p-4 rounded-lg border border-gray-200 flex justify-center">
                        <img
                          src={`data:image/png;base64,${moleculeImageBase64}`}
                          alt="Molecule Structure"
                          className="max-w-full h-auto shadow-sm"
                        />
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </main>
          
          {/* Footer */}
          <footer className="py-6 px-10 text-center text-gray-500 border-t border-gray-200 text-sm">
            <div>Powered by Advanced Machine Learning for Protein-SMILES Conversion</div>
          </footer>
        </div>
      </div>
    </div>
  );
}