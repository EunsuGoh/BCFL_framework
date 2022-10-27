import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';

function Plot({ flresult }) {
  // 라운드별 토큰 정보 받기
  const tokenPerRounds = flresult.filter((elem) => {
    if (elem.includes('Scores in round')) return true;
    else return false;
  });

  // console.log(tokenPerRounds);
  // 라운드별 토큰값
  let tokenPerRound_token = tokenPerRounds.map((elem) => elem.split(':'));
  tokenPerRound_token = tokenPerRound_token.map((elem) => {
    return elem.filter((item) => {
      if (item.includes('0.')) return true;
      else return false;
    });
  });
  // tokenPerRound_token = tokenPerRound_token.slice(
  //   1,
  //   tokenPerRound_token.length - 1
  // );
  // console.log(tokenPerRound_token);
  tokenPerRound_token = tokenPerRound_token.map((elem) => {
    return elem.map((item) => {
      return item.slice(1, item.length - 1);
    });
  });
  // tokenPerRound_token = tokenPerRound_token.map((elem) => {
  //   return elem.split(', ');
  // });
  // tokenPerRound_token = tokenPerRound_token[0].concat(tokenPerRound_token[1]);
  let tokenPerRound_token_0 = tokenPerRound_token[0][0].split(', ');
  let tokenPerRound_token_1 = tokenPerRound_token[1][0].split(', ');
  // console.log(tokenPerRound_token_0);
  // console.log(tokenPerRound_token_1);
  tokenPerRound_token = tokenPerRound_token_0.concat(tokenPerRound_token_1);

  // 라운드별 cid값
  let tokenPerRound_cid = tokenPerRounds.map((elem) => elem.split(':'));
  tokenPerRound_cid = tokenPerRound_cid.map((elem) => {
    return elem.filter((item) => {
      if (item.includes('Qm')) return true;
      else return false;
    });
  });

  tokenPerRound_cid = tokenPerRound_cid.map((elem) => {
    return elem.map((item) => {
      return item.slice(1, item.length - 1);
    });
  });

  let tokenPerRound_cid_0 = tokenPerRound_cid[0][0].split(', ');
  let tokenPerRound_cid_1 = tokenPerRound_cid[1][0].split(', ');

  tokenPerRound_cid = tokenPerRound_cid_0.concat(tokenPerRound_cid_1);
  tokenPerRound_cid = tokenPerRound_cid.map((elem) => {
    return elem.slice(1, elem.length - 1);
  });

  // console.log(tokenPerRound_token); // 라운드별 토큰
  // console.log(tokenPerRound_cid); // 위 토큰에 해당하는 cid

  let token = [];
  let cid = [];

  // let tokenSplit_1 = tokenPerRound_token[0][0].split(', ');
  // let tokenSplit_2 = tokenPerRound_token[1][0].split(', ');

  // token = tokenSplit_1.concat(tokenSplit_2);

  // token = token.map((elem) => {
  //   if (elem.includes('[')) return elem.slice(1);
  //   else if (elem.includes(']')) return elem.slice(0, elem.length);
  //   else {
  //     return elem;
  //   }
  // });
  // console.log(token);
  // console.log(tokenSplit_2);
  // console.log(tokenPerRound_cid);

  // 라운드별 cid 정보 받기
  const cidsPerRounds = flresult.filter((elem) => {
    if (elem.includes('local model cid')) return true;
    else return false;
  });
  let cidsPerRoundsSplit = cidsPerRounds.map((elem) => {
    return elem.split(',');
  });
  cidsPerRoundsSplit = cidsPerRoundsSplit.map((elem) => {
    return (elem = elem.map((item) => {
      if (item.includes('Adding')) {
        let name = item.split(':');
        return name[0];
      } else if (item.includes('model cid')) {
        let cid = item.split(': ');
        return cid[1];
      } else {
        let round = item.split(': ');
        return round[1];
      }
    }));
  });

  for (let i = 0; i < cidsPerRoundsSplit.length; i++) {
    cidsPerRoundsSplit[i].push(tokenPerRound_token[i]);
  }
  console.log(cidsPerRoundsSplit);
  const roundData = [
    {
      name: 'Bob',
      round1: 4000,
      round2: 2400,
      amt: 2400,
    },
    {
      name: 'Charlie',
      round1: 3000,
      round2: 1398,
      amt: 2210,
    },
    {
      name: 'David',
      round1: 2000,
      round2: 9800,
      amt: 2290,
    },
    {
      name: 'Eve',
      round1: 2780,
      round2: 3908,
      amt: 2000,
    },
  ];

  for (let j = 0; j < cidsPerRoundsSplit.length; j++) {
    if (
      cidsPerRoundsSplit[j][0] === 'Bob' &&
      cidsPerRoundsSplit[j][2] === '1'
    ) {
      roundData[0].round1 = cidsPerRoundsSplit[j][3];
    }
    if (
      cidsPerRoundsSplit[j][0] === 'Bob' &&
      cidsPerRoundsSplit[j][2] === '2'
    ) {
      roundData[0].round2 = cidsPerRoundsSplit[j][3];
    }
    if (
      cidsPerRoundsSplit[j][0] === 'Charlie' &&
      cidsPerRoundsSplit[j][2] === '1'
    ) {
      roundData[1].round1 = cidsPerRoundsSplit[j][3];
    }
    if (
      cidsPerRoundsSplit[j][0] === 'Charlie' &&
      cidsPerRoundsSplit[j][2] === '2'
    ) {
      roundData[1].round2 = cidsPerRoundsSplit[j][3];
    }
    if (
      cidsPerRoundsSplit[j][0] === 'David' &&
      cidsPerRoundsSplit[j][2] === '1'
    ) {
      roundData[2].round1 = cidsPerRoundsSplit[j][3];
    }
    if (
      cidsPerRoundsSplit[j][0] === 'David' &&
      cidsPerRoundsSplit[j][2] === '2'
    ) {
      roundData[2].round2 = cidsPerRoundsSplit[j][3];
    }
    if (
      cidsPerRoundsSplit[j][0] === 'Eve' &&
      cidsPerRoundsSplit[j][2] === '1'
    ) {
      roundData[3].round1 = cidsPerRoundsSplit[j][3];
    }
    if (
      cidsPerRoundsSplit[j][0] === 'Eve' &&
      cidsPerRoundsSplit[j][2] === '2'
    ) {
      roundData[3].round2 = cidsPerRoundsSplit[j][3];
    }
  }
  // 최종 토큰 정보
  const FinalTokenResult = flresult.slice(flresult.length - 4);
  // console.log(FinalTokenResult);
  const Bobinfo = FinalTokenResult[0].split(' ');
  const BobToken = Bobinfo[2] / 1000000000000;

  const Charlieinfo = FinalTokenResult[1].split(' ');
  const CharlieToken = Charlieinfo[2] / 1000000000000;

  const Davidinfo = FinalTokenResult[2].split(' ');
  const DavidToken = Davidinfo[2] / 1000000000000;

  const Eveinfo = FinalTokenResult[3].split(' ');
  const EveToken = Eveinfo[2] / 1000000000000;

  const FinalToken = {
    Bob: BobToken,
    Charlie: CharlieToken,
    David: DavidToken,
    Eve: EveToken,
  };
  const finalData = [
    {
      name: 'Bob',
      token: FinalToken.Bob,
    },
    {
      name: 'Charlie',
      token: FinalToken.Charlie,
    },
    {
      name: 'David',
      token: FinalToken.David,
    },
    {
      name: 'Eve',
      token: FinalToken.Eve,
    },
  ];

  return (
    <div
      className='Plot'
      style={{
        display: 'flex',
        justifyContent: 'center',
        margin: '1rem',
      }}
    >
      <div className='Con_Per_round'>
        <div
          style={{
            textAlign: 'center',
          }}
        >
          Contribution Score per Rounds
        </div>
        <BarChart
          width={500}
          height={300}
          data={roundData}
          margin={{
            top: 5,
            right: 30,
            left: 20,
            bottom: 5,
          }}
        >
          <CartesianGrid strokeDasharray='3 3' />
          <XAxis dataKey='name' />
          <YAxis />
          <Tooltip />
          <Legend />
          <Bar dataKey='round1' fill='#8884d8' />
          <Bar dataKey='round2' fill='#82ca9d' />
        </BarChart>
      </div>

      <div className='Con_Final'>
        <div
          style={{
            textAlign: 'center',
          }}
        >
          Final Contributivity (token)
        </div>
        <div>
          <BarChart
            width={500}
            height={300}
            data={finalData}
            margin={{
              top: 5,
              right: 30,
              left: 20,
              bottom: 5,
            }}
          >
            <CartesianGrid strokeDasharray='3 3' />
            <XAxis dataKey='name' />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey='token' fill='#8884d8' />
          </BarChart>
        </div>
      </div>
    </div>
  );
}

export default Plot;
