// custom_crowdsource.py 연결
const { spawn } = require('child_process'); // run python script
const fs = require('fs');
const ciqlJson = require('ciql-json'); // 기능참고 : https://www.npmjs.com/package/ciql-json
const { resolveSoa } = require('dns');
const pythonShell = require('python-shell');
const PYTHON_SCRIPT_PATH =
  '/home/dy/2cp_workspace/2CP/crowdsource_back/back/src/utils/custom_crowdsource.py';
const EVALUATOR = 'Alice';
const TRAINERS = ['Bob', 'Charlie', 'David', 'Eve'];
const axios = require("axios")

// post : 참가자 get 요청
const showStakeholers = async (req, res) => {
  res.status(200).send({
    msg: 'success',
    evaluator: EVALUATOR,
    trainers: TRAINERS,
  });
};

// post : 연합학습 시작
const flStart = async (req, res) => {
  try {
    let options = {
      mode: 'text',
      pythonPath: '',
      pythonOptions: ['-u'],
      scriptPath: '',
      args: ['1'],
    };

    await pythonShell.PythonShell.run(
      PYTHON_SCRIPT_PATH,
      options,
      function (err, result) {
        if (err) {
          console.log(err);
          res.status(400).send({ msg: err });
        } else {
          console.log('results : %j', result);
          console.log('finished');
          res.status(200).send({ msg: 'success', result: result });
        }
      }
    );
  } catch (e) {
    res.status(400).send({ msg: e });
  }
};

module.exports = {
  showStakeholers,
  flStart
};
