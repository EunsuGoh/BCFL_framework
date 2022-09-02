
pragma solidity >=0.4.21 <0.7.0;


contract Modelcids {

    string cid;


    function store(string memory local_cid) public {
        cid = local_cid;
    }

 
    function retrieve() public view returns (string memory){
        return cid;
    }
}