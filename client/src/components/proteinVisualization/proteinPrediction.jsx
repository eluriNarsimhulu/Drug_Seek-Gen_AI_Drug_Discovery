// import React, { useState } from 'react';
// import { Container, Row, Col, Button, Form, Alert } from 'react-bootstrap';
// import ProteinViewer from './proteinViewer';
// import './proteinPrediction.css';
// import * as $3Dmol from '3dmol';

// function ProteinPrediction() {
//   const [sequence, setSequence] = useState(
//     "MGSSHHHHHHSSGLVPRGSHMRGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTLIFACENDSIAPVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQALIGKKGVAWMKRFMDNDTRYSTFACENPNSTRVSDFRTANCSLEDPAANKARKEAELAAATAEQ"
//   );
//   const [pdbData, setPdbData] = useState(null);
//   const [loading, setLoading] = useState(false);
//   const [error, setError] = useState(null);

//   const calculatePlDDT = (pdbString) => {
//     const lines = pdbString.split('\n');
//     let sum = 0;
//     let count = 0;

//     lines.forEach(line => {
//       if (line.startsWith('ATOM')) {
//         const bFactor = parseFloat(line.substring(60, 66).trim());
//         sum += bFactor;
//         count++;
//       }
//     });

//     return count > 0 ? (sum / count).toFixed(4) : 0;
//   };

//   const handlePredict = async () => {
//     setLoading(true);
//     setError(null);

//     try {
//       if (!sequence || sequence.length < 10) {
//         throw new Error('Please enter a valid protein sequence (minimum 10 characters)');
//       }

//       const response = await fetch('http://localhost:8080/api/protein/fold', {
//         method: 'POST',
//         headers: {
//           'Content-Type': 'application/json',
//         },
//         body: JSON.stringify({ sequence }),
//       });

//       if (!response.ok) {
//         throw new Error('Prediction failed - server error');
//       }

//       const data = await response.json();
//       if (!data.pdb) {
//         throw new Error('Invalid PDB data received');
//       }
//       setPdbData(data.pdb);
//     } catch (err) {
//       setError(err.message);
//       console.error('Prediction error:', err);
//     } finally {
//       setLoading(false);
//     }
//   };

//   const downloadPdb = () => {
//     const element = document.createElement('a');
//     const file = new Blob([pdbData], { type: 'text/plain' });
//     element.href = URL.createObjectURL(file);
//     element.download = 'predicted.pdb';
//     document.body.appendChild(element);
//     element.click();
//     document.body.removeChild(element);
//   };

//   return (
//     <Container fluid>
//       <Row>
//         <Col md={3} className="sidebar bg-light p-4">
//           <h1>🎈 ESMFold</h1>
//           <p>
//             <a href="https://esmatlas.com/about" target="_blank" rel="noopener noreferrer">ESMFold</a> is an end-to-end single sequence protein structure predictor.
//           </p>

//           <Form.Group controlId="sequenceInput" className="mb-3">
//             <Form.Label>Input sequence</Form.Label>
//             <Form.Control
//               as="textarea"
//               rows={15}
//               value={sequence}
//               onChange={(e) => setSequence(e.target.value)}
//               placeholder="Enter protein sequence..."
//             />
//           </Form.Group>

//           <Button
//             variant="primary"
//             onClick={handlePredict}
//             disabled={loading}
//             className="w-100"
//           >
//             {loading ? 'Predicting...' : 'Predict'}
//           </Button>
//         </Col>

//         <Col md={9} className="main-content p-4">
//           {error && <Alert variant="danger">{error}</Alert>}

//           {!sequence && (
//             <Alert variant="warning">👈 Enter protein sequence data!</Alert>
//           )}

//           {pdbData && (
//             <>
//               <div className="center-text-content mt-4">
//                 <h2>Visualization of predicted protein structure</h2>
//               </div>
//               <div className="protein-viewer-container">
//                 <ProteinViewer pdbData={pdbData} />
//               </div>

//               <div className="center-text-content mt-4 p-3 bg-light rounded">
//                 <h2>plDDT</h2>
//                 <p>plDDT is a per-residue estimate of the confidence in prediction on a scale from 0-100.</p>
//                 <p className="plddt-value">plDDT: {calculatePlDDT(pdbData)}</p>

//                 <Button variant="success" onClick={downloadPdb} className="mt-2">
//                   Download PDB
//                 </Button>
//               </div>
//             </>
//           )}
//         </Col>
//       </Row>
//     </Container>
//   );
// }

// export default ProteinPrediction;



import React, { useState, useEffect, useRef } from 'react';

const ProteinViewer = ({ pdbData }) => {
  const viewerRef = useRef(null);

  useEffect(() => {
    if (!pdbData || !window.$3Dmol) return;
    
    // Clear any existing content
    if (viewerRef.current) {
      viewerRef.current.innerHTML = '';
    }

    // Set a short timeout to ensure DOM is ready
    setTimeout(() => {
      if (!viewerRef.current) return;
      
      // Create viewer directly in the ref element (no new element creation)
      const viewer = window.$3Dmol.createViewer(viewerRef.current, {
        backgroundColor: '#111827',
        width: '100%',
        height: '100%'
      });
      
      try {
        viewer.addModel(pdbData, 'pdb');
        viewer.setStyle({ cartoon: { color: 'spectrum' } });
        viewer.zoomTo();
        viewer.zoom(1.8, 800);
        viewer.spin(true);
        viewer.render();
      } catch (error) {
        console.error('Error rendering protein:', error);
      }
    }, 100);

    return () => {
      if (viewerRef.current) {
        viewerRef.current.innerHTML = '';
      }
    };
  }, [pdbData]);

  return (
    <div className="w-full h-96 rounded-xl overflow-hidden border-4 border-indigo-100 shadow-xl" style={{ position: 'relative' }}>
      <div ref={viewerRef} className="w-full h-full absolute inset-0" style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0 }} />
    </div>
  );
};

function ProteinPrediction() {
  const [sequence, setSequence] = useState(
    "MGSSHHHHHHSSGLVPRGSHMRGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTLIFACENDSIAPVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQALIGKKGVAWMKRFMDNDTRYSTFACENPNSTRVSDFRTANCSLEDPAANKARKEAELAAATAEQ"
  );
  const [pdbData, setPdbData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const calculatePlDDT = (pdbString) => {
    const lines = pdbString.split('\n');
    let sum = 0;
    let count = 0;

    lines.forEach(line => {
      if (line.startsWith('ATOM')) {
        const bFactor = parseFloat(line.substring(60, 66).trim());
        sum += bFactor;
        count++;
      }
    });

    return count > 0 ? (sum / count).toFixed(4) : 0;
  };

  const handlePredict = async () => {
    setLoading(true);
    setError(null);

    try {
      if (!sequence || sequence.length < 10) {
        throw new Error('Please enter a valid protein sequence (minimum 10 characters)');
      }

      const response = await fetch('http://localhost:8080/api/protein/fold', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ sequence }),
      });

      if (!response.ok) {
        throw new Error('Prediction failed - server error');
      }

      const data = await response.json();
      if (!data.pdb) {
        throw new Error('Invalid PDB data received');
      }
      setPdbData(data.pdb);
    } catch (err) {
      setError(err.message);
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  const downloadPdb = () => {
    const element = document.createElement('a');
    const file = new Blob([pdbData], { type: 'text/plain' });
    element.href = URL.createObjectURL(file);
    element.download = 'predicted.pdb';
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
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
    <div className="min-h-screen py-10 px-4">
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
      <div className="max-w-7xl mx-auto bg-white rounded-3xl shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="bg-gradient-to-r from-purple-600 to-indigo-700 text-white p-8 relative overflow-hidden">
          <div className="absolute inset-0 opacity-10 bg-pattern"></div>
          <div className="relative z-10">
            <h1 className="text-4xl font-extrabold tracking-tight">ESMFold Protein Structure Predictor</h1>
            <p className="text-lg opacity-80 mt-2">End-to-end single sequence protein structure prediction</p>
          </div>
        </div>

        <div className="flex flex-col lg:flex-row">
          {/* Sidebar */}
          <div className="lg:w-1/3 p-6 bg-gray-50 border-r border-gray-200">
            <div className="mb-6">
              <h2 className="text-xl font-bold text-purple-800 flex items-center">
                <span className="text-2xl mr-2">🧬</span> Input Sequence
              </h2>
              <p className="text-gray-600 text-sm mb-4">
                Enter a protein sequence to predict its 3D structure using{" "}
                <a href="https://esmatlas.com/about" target="_blank" rel="noopener noreferrer" className="text-indigo-600 hover:text-indigo-800 underline">
                  ESMFold
                </a>
              </p>
            </div>

            <div className="mb-6">
              <label htmlFor="sequence" className="block text-sm font-medium text-gray-700 mb-2">
                Protein Sequence:
              </label>
              <textarea
                id="sequence"
                rows="12"
                className="w-full px-4 py-3 rounded-xl border-2 border-gray-300 focus:border-indigo-500 focus:ring focus:ring-indigo-200 focus:ring-opacity-50 font-mono text-sm resize-none"
                placeholder="Enter protein sequence..."
                value={sequence}
                onChange={(e) => setSequence(e.target.value)}
              ></textarea>
            </div>

            <button
              onClick={handlePredict}
              disabled={loading}
              className={`w-full py-3 px-4 rounded-xl font-medium text-white transition-all duration-300 shadow-md flex items-center justify-center ${
                loading
                  ? "bg-gray-400 cursor-not-allowed"
                  : "bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 hover:-translate-y-1"
              }`}
            >
              {loading ? (
                <>
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Processing...
                </>
              ) : (
                "Predict Structure"
              )}
            </button>
          </div>

          {/* Main Content */}
          <div className="lg:w-2/3 p-6">
            {error && (
              <div className="bg-red-50 border-l-4 border-red-500 p-4 mb-6 rounded-lg">
                <div className="flex items-center">
                  <div className="flex-shrink-0">
                    <svg className="h-5 w-5 text-red-500" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                    </svg>
                  </div>
                  <div className="ml-3">
                    <p className="text-sm text-red-700">{error}</p>
                  </div>
                </div>
              </div>
            )}

            {!sequence && (
              <div className="bg-yellow-50 border-l-4 border-yellow-500 p-4 rounded-lg">
                <div className="flex items-center">
                  <div className="flex-shrink-0">
                    <svg className="h-5 w-5 text-yellow-500" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2h-1V9a1 1 0 00-1-1z" clipRule="evenodd" />
                    </svg>
                  </div>
                  <div className="ml-3">
                    <p className="text-sm text-yellow-700">👈 Enter protein sequence data!</p>
                  </div>
                </div>
              </div>
            )}

            {pdbData && (
              <div className="space-y-8">
                <div className="text-center">
                  <h2 className="text-2xl font-bold text-gray-800 mb-2">Visualization of Predicted Protein Structure</h2>
                  <p className="text-gray-600 text-sm">3D rendering of the protein structure based on ESMFold prediction</p>
                </div>

                <div className="protein-viewer-container relative w-full">
                  <ProteinViewer pdbData={pdbData} />
                </div>

                <div className="bg-gray-50 rounded-2xl p-6 text-center shadow-md border border-gray-200">
                  <h2 className="text-2xl font-bold text-indigo-800 mb-3">plDDT Score</h2>
                  <p className="text-gray-600 mb-4">
                    plDDT is a per-residue estimate of the confidence in prediction on a scale from 0-100.
                  </p>
                  <div className="inline-block px-6 py-3 rounded-lg bg-indigo-100 font-mono text-xl font-bold text-indigo-800 mb-4">
                    {calculatePlDDT(pdbData)}
                  </div>
                  <div>
                    <button
                      onClick={downloadPdb}
                      className="inline-flex items-center px-5 py-2 rounded-lg bg-green-600 hover:bg-green-700 text-white font-medium transition-colors duration-300"
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                      </svg>
                      Download PDB
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="border-t border-gray-200 py-4 px-6 text-center text-gray-500 text-sm">
          Powered by ESMFold Protein Structure Prediction Technology
        </div>
      </div>
    </div>
  );
}

export default ProteinPrediction;