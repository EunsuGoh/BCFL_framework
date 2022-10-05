// custom_crowdsource.py 연결
const { spawn } = require('child_process'); // run python script
const fs = require('fs');
const ciqlJson = require('ciql-json'); // 기능참고 : https://www.npmjs.com/package/ciql-json
const { resolveSoa } = require('dns');
const pythonShell = require('python-shell');
let PYTHON_SCRIPT_PATH ="";
const EVALUATOR = 'evaluator';
const TRAINERS = ['Bob', 'Charlie', 'David', 'Eve'];
const axios = require("axios")

//로그인
const login = async (req,res) =>{
  const id = req.body.id
  const password = req.body.password

  if(id==='evaluator'){
    res.status(200).send({
      msg : "Hello, Evaluator",
      tag : "eval"
    })
  }else{
    res.status(200).send({
      msg : "Hello, Trainer",
      tag : "train"
    })
  }
}

// 트레이닝 태스크 전송
const showFlTask = async (req,res)=>{
  let curFlTask1 = await ciqlJson.open("./src/eval_Fltask.json").getData()
  res.status(200).send({msg:"success",data:curFlTask1})
}

// 학습 참여요청
const train_participate = async (req,res)=>{
  let curFlTask = await ciqlJson.open("./src/eval_Fltask.json").getData()
  let FlTask = curFlTask.evaluator
  let trainerNum = parseInt(FlTask.trainers)
  if(trainerNum === parseInt( FlTask.maxNumUpdates)){
    res.status(200).send({msg : "You can no longer participate. Wait for new train recruitment", flag : "F"})
  }
  else{
    trainerNum+=1;
    FlTask.trainers = String(trainerNum);
    const result = await ciqlJson.open("./src/eval_Fltask.json").set("evaluator",FlTask).save()
    res.status(200).send({msg:`trainer number setting success : ${FlTask.trainers}`, train_idx : `${String(trainerNum)}`})
  }
}

// 평가자의 학습시작 여부 체크
const train_checkTrainFlag = async (req,res) =>{
  let curFlTask = await ciqlJson.open("./src/eval_Fltask.json").getData()
  let trainingFlag = curFlTask.evaluator.trainingFlag
  if(trainingFlag === 'True'){
    res.status(200).send({msg : "True"})
  }else{
    res.status(200).send({msg : "False"})
  }
}

// 평가자의 학습시작
const train_startTrain = async (req,res)=>{
  let curFlTask = await ciqlJson.open("./src/eval_Fltask.json").getData()  
  let genesisSetting = curFlTask.evaluator
  const trainer_idx = req.body.index
  // start evaluator's fl process
  if (genesisSetting.model === 'cifar'){
    PYTHON_SCRIPT_PATH = __dirname+'/utils/pytorch_cifar10/pytorch_cifar10_eval.py'
  }
  else if (genesisSetting.model === 'mnist'){
    PYTHON_SCRIPT_PATH = __dirname+'/utils/pytorch_mnist/pytorch_mnist_eval.py'
  }
  else if (genesisSetting.model === 'femnist'){
    PYTHON_SCRIPT_PATH = __dirname+'/utils/pytorch_femnist/pytorch_femnist_eval.py'
  }
  else{
      PYTHON_SCRIPT_PATH = __dirname+'/utils/pytorch_shakespeare/pytorch_shakespeare_train.py'
  }
  try{
    let options = {
      mode: 'text',
      pythonPath: '',
      pythonOptions: ['-u'],
      scriptPath: '',
      args: [trainer_idx, genesisSetting.globalRounds],
    };
    console.log(PYTHON_SCRIPT_PATH)
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
  }catch(e){
    res.status(500).send({msg:"Server Error"})
  }
}
module.exports = {
  login,
  showFlTask,
  train_participate,
  train_checkTrainFlag,
  train_startTrain
};
