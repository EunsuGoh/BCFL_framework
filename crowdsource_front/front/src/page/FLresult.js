import { useState, useEffect } from 'react';

import axios from 'axios';
import serverEndpoint from '../server_setting.json';
import loading from '/home/dy/2cp_new/crowdsource_front/front/src/kOnzy.gif';
import ResultComponent from '../components/ResultComponent';

function FLresult() {
  const [flresult, setFLResult] = useState([]);
  const listResults = flresult.map((result) => <li>{result}</li>);
  const startFL = async () => {
    try {
      const result = await axios.post(serverEndpoint.server + '/flStart');
      // console.log(result.data.result);
      setFLResult(result.data.result);
    } catch (e) {
      console.log(e);
    }
  };
  useEffect(() => {
    startFL();
  }, []);

  return (
    <div className='FLresult'>
      <div
        className='trainingResult'
        style={{
          textAlign: 'center',
        }}
      >
        Training Result(Full)
        <br />
        <div
          style={{
            display: 'flex',
            justifyContent: 'center',
          }}
        >
          <div
            style={{
              border: '2px solid darkgrey',
              width: '50%',
              borderStyle: 'dotted',
              textAlign: 'left',
            }}
          >
            <ul>
              <li>Evaluator : Alice</li>
              <li>Large Data Trainers : Bob</li>
              <li>Medium Data Trainers : Charlie</li>
              <li>Small Data Trainers : David, Eve</li>
            </ul>
          </div>
        </div>
        {flresult.length !== 0 ? (
          <ResultComponent flresult={flresult} />
        ) : (
          <img
            src={loading}
            alt='loading'
            style={{
              width: '10%',
              height: '10%',
            }}
          />
        )}
      </div>
    </div>
  );
}

export default FLresult;
