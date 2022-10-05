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

// 평가자 : 트레이닝 세팅
const eval_setGenesis = async (req,res) =>{
  // Genesis setting & register json file
  const {globalRounds,maxNumUpdates,model} = req.body.genesisSetting
 try{ await ciqlJson
            .create({})
            .set("evaluator",{"model":model,"globalRounds":String(globalRounds), "maxNumUpdates":String(maxNumUpdates), "trainers":"0","trainingFlag":"False"})
            .save("./src/eval_Fltask.json")
            res.status(200).send({msg:"Success"})
          }catch(e){
              res.status(400).send({e})
            }
};

// 트레이닝 태스크 전송
const showFlTask = async (req,res)=>{
  let curFlTask1 = await ciqlJson.open("./src/eval_Fltask.json").getData()
  res.status(200).send({msg:"success",data:curFlTask1})
}

// 평가자의 학습시작 여부 등록
const eval_setFlag = async(req,res)=>{
  let curFlTask = await ciqlJson.open("./src/eval_Fltask.json").getData()  
  let genesisSetting = curFlTask.evaluator
  genesisSetting.trainingFlag = "True"
  await ciqlJson.open("./src/eval_Fltask.json").set("evaluator",genesisSetting).save()

  res.status(200).send({msg:"success"})
}

// 평가자 : 연합학습 초기세팅 + 실행
const eval_startEval = async (req,res)=>{
  let curFlTask = await ciqlJson.open("./src/eval_Fltask.json").getData()  
  let genesisSetting = curFlTask.evaluator

  if(genesisSetting.trainers !== genesisSetting.maxNumUpdates){
    res.status(200).send({msg : "Cannot Start FL yet, You should wait for more trainers", "flag":"F"})
  }else{
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
       PYTHON_SCRIPT_PATH = __dirname+'/utils/pytorch_shakespeare/pytorch_shakespeare_eval.py'
    }
    try{
      let options = {
        mode: 'text',
        pythonPath: '',
        pythonOptions: ['-u'],
        scriptPath: '',
        args: [genesisSetting.maxNumUpdates, genesisSetting.globalRounds],
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

}




module.exports = {
  login,
  eval_setGenesis,
  showFlTask,
  eval_setFlag,
  eval_startEval,

};
