import { useState, useEffect } from 'react';
import Plot from './Plot';

function ResultComponent({ flresult }) {
  // 3. 학습 과정
console.log(flresult)
  const listFlResults = flresult.map((elem) => {
    return <li>{elem}</li>;
  });
  // console.log(flResults);
  return (
    <div
      className='ResultComponent'
      style={{
        display: 'flex',
        width: '100%',
        height: '100%',
        justifyContent: 'center',
        alignItems: 'center',
        flexDirection: 'column',
      }}
    >   
      <div
        className='Training'
        style={{
          background: 'white',
          borderRadius: '10px',
          padding: '2%',
          border: '2px solid darkblue',
          width: '70%',
        }}
      >
        <div
          style={{
            textAlign: 'left',
            fontSize: '1.2rem',
          }}
        >
          Federated Learning process
          <div
            style={{
              textAlign: 'left',
              fontSize: '1rem',
              overflow: 'auto',
              height: '200px',
            }}
          >
            <ul>{listFlResults}</ul>
          </div>
        </div>
      </div>{' '}
      {/* <div className='plotArea'>
        <Plot flresult={flresult} />
      </div> */}
    </div>
  );
}

export default ResultComponent;
