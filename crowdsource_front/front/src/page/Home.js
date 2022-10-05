import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import serverEndpoint from '../server_setting.json';
import Box from '@mui/material/Box';
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button';
import Typography from '@mui/material/Typography';


function Home() {
  const navigate = useNavigate();
  const [id, setId] = useState("");
  const [password, setPassword] = useState("");
  const [tag,setTag] = useState("train");

  useEffect(()=>{
    console.log(id)
    console.log(password)
  },[id,password])
  const idHandler = (e)=>{
    setId(e.target.value)
  }

  const login = async () =>{
    try{
      const result = await axios.post(serverEndpoint.server+"/login",{id,password})
      setTag(result.data.tag)
      window.alert(result.data.msg)
      navigate('/Main',{state:{
        tag:result.data.tag
      }})

    }catch(e){
      console.log(e);
    }
  }

  const passwordHandler = (e)=>{
    setPassword(e.target.value)
  }
  return (
    <div className='Home' style ={{
      display :"flex",
      flexDirection : "column",
      justifyContent:"center",
      alignItems:"center"
    }}>

      <Box sx={{
          '& > :not(style)': { m: 1, width: '25ch' },
          display:"flex",
          flexDirection : "column",
          justifyContent:"center",
          alignItems:"center",
          height:"50%"
        }}
        >
         <Typography variant="h5" component="div">
           Welcome, This is cross-device front page
        </Typography>
        <Typography variant="h5" component="div">
        You should log in first
        </Typography>
      </Box>

      <Box component = "form"
        sx={{
          '& > :not(style)': { m: 1, width: '25ch' },
          display:"flex",
          flexDirection : "column",
          justifyContent:"center",
          alignItems:"center"
        }}
        noValidate
        autoComplete="off"
        >
          <TextField id="outlined-basic" label="id" variant="outlined" onChange={idHandler} />
          <br/>
          <TextField id="outlined-basic" label="password" variant="outlined" type = "password" onChange={passwordHandler}/>
          <br/>
          <Button variant="text" sx={{
            width:"100%"
          }} onClick={login}>submit</Button>

      </Box>
      <br/>
    </div>
  );
}

export default Home;
