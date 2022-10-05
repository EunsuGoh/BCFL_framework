import { useState, useEffect } from 'react';
import Plot from './Plot';

function ResultComponent({ flresult }) {
  // console.log(listResults);
  // 1. 컨트랙트 연결 정보 받아오기
  const contractConnect = flresult.filter((elem) => {
    if (elem.includes('0x')) return true;
    else return false;
  });
  const contAddrSplit = contractConnect[0].split(' ');
  const contAddr = contAddrSplit[contAddrSplit.length - 1];
  const listContInfo = contractConnect.map((elem) => {
    return <li>{elem}</li>;
  });

  // 2. 제네시스 모델 세팅 정보 받아오기
  const genesisSetting = flresult.filter((elem) => {
    if (elem.includes('Genesis') || elem.includes('genesis')) return true;
    else return false;
  });
  const listgenesisSetting = genesisSetting.map((elem) => {
    return <li>{elem}</li>;
  });

  // 3. 학습 과정
  const flResults = flresult.slice(9);
  const listFlResults = flResults.map((elem) => {
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
        className='ContractInfo'
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
          1. Contract Address : {contAddr}
          <div
            style={{
              textAlign: 'left',
              fontSize: '1rem',
            }}
          >
            <ul>{listContInfo}</ul>
          </div>
        </div>
      </div>
      <div
        className='SetGenesis'
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
          2. Setting Genesis
          <div
            style={{
              textAlign: 'left',
              fontSize: '1rem',
            }}
          >
            <ul>{listgenesisSetting}</ul>
          </div>
        </div>
      </div>
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
          3. Federated Learning
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
      <div className='plotArea'>
        <Plot flresult={flresult} />
      </div>
    </div>
  );
}

export default ResultComponent;
