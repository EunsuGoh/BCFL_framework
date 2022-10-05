import axios from 'axios';
import { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import Button from '@mui/material/Button';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import serverEndpoint from '../server_setting.json';
import TextField from '@mui/material/TextField';
import Select, { SelectChangeEvent } from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';



function CreateFLTask() {

  const [globalRounds, setGlobalRounds] = useState(0);
  const [maxNumUpdates, setmaxNumUpdates] = useState(0);
  const [model,setModel] = useState("");

  

  useEffect(()=>{
    console.log(globalRounds)
    console.log(maxNumUpdates)
    console.log(model)
  },[globalRounds,maxNumUpdates,model])


  const handleglobalRounds = (e)=>{
    setGlobalRounds(e.target.value)
  }
  const handlemaxNumUpdates = (e)=>{
    setmaxNumUpdates(e.target.value)
  }
  const handleModel = (e) => {
    setModel(e.target.value);
  };

  const setGenesisModel = async ()=>{
    const result = await axios.post(serverEndpoint.server+"/eval_setGenesis",{
      genesisSetting:{
        globalRounds,maxNumUpdates,model
      }
    })
    window.alert(result.data.msg)
  }
  const showFlList = ()=>{
    navigate("/FLlist",{state:{
      tag : location.state.tag
    }
    })
  }

  const navigate = useNavigate();
  const location = useLocation()
  
  return (
    <div className='createFLTask' style={{
      display:"flex",
      flexDirection:"column",
      alignItems:"center",
      justifyContent:"cneter"
    }}>
      
      <Box sx={{
        display:"flex",
        flexDirection:"column",
        alignItems:"center",
        justifyContent:"cneter"
      }}>
      <Typography variant="h5" component="div">
        Global Genesis Setting
        </Typography>
      <TextField id="outlined-basic" label="Global Rounds" variant="outlined" onChange={handleglobalRounds}/>
      <TextField id="outlined-basic" label="Max number of updates" variant="outlined" onChange={handlemaxNumUpdates}/>
      <Select
          labelId="demo-simple-select-label"
          id="demo-simple-select"
          value={model}
          label="model"
          onChange={handleModel}
        >
          <MenuItem value={"cifar"}>cifar-cnn</MenuItem>
          <MenuItem value={"mnist"}>mnist-cnn</MenuItem>
          <MenuItem value={"femnist"}>femnist-cnn</MenuItem>
          <MenuItem value={"shakespeare"}>shakespeare-lstm</MenuItem>
        </Select>
      <Button variant="text" sx={{
            width:"100%"
          }} onClick={setGenesisModel}>Register</Button>
      <Button variant="text" sx={{
            width:"100%"
          }} onClick={showFlList}>Show FL Lists</Button>
      </Box>
    </div>
  );
}

export default CreateFLTask;
