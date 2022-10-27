const pythonShell = require('python-shell');
const PYTHON_SCRIPT_PATH ='/home/dy/2cp_workspace/2CP/crowdsource_front/front/src/utils/flower_client.py';

const startFlowerClient = async ()=>{
  let options = {
    mode: 'text',
    pythonPath: '',
    pythonOptions: ['-u'],
    scriptPath: '',
    args: ['1'],
  };

  try{
    await pythonShell.PythonShell.run(
      PYTHON_SCRIPT_PATH,
      options,
      function (err, result) {
        if (err) {
          console.log(err);
        } else {
          console.log("Flower client starting ... ")
          console.log('results : %j', result);
        }
      }
    );
  }catch(e){
    console.log(e);
    return e;
  }
  
}
module.exports = {
  startFlowerClient
};