// pragma solidity >=0.4.21 <0.7.0;
pragma solidity ^0.8.6;

import "./Crowdsource.sol";

/// @title Deploys and manages Crowdsourcing auxiliaries that make up a Consortium Federated Learning process.
/// @author Harry Cai
contract Consortium {
    address internal mainAddress;

    address[] internal auxAddresses;

    bytes32 internal genesis;

    uint256 internal roundDuration;

    uint256 internal numTrainers;

    address internal account;

    uint256 internal round;

    address internal contractAddress;

    address internal evaluator;

    constructor() public {
        Crowdsource main = new Crowdsource();
        mainAddress = address(main);
    }

    /// @return The address of the Crowdsourcing main.
    function main() external view returns (address) {
        return mainAddress;
    }

    /// @return The addresses of the Crowdsourcing auxiliaries.
    function auxiliaries() external view returns (address[] memory) {
        return auxAddresses;
    }

    /// @return round The index of the latest training round among Crowdsourcing auxiliaries and main.
    function latestRound() external view returns (uint256 round) {
        Crowdsource main = Crowdsource(mainAddress);
        round = main.currentRound();
        for (uint256 i = 0; i < auxAddresses.length; i++) {
            Crowdsource aux = Crowdsource(auxAddresses[i]);
            uint256 auxRound = aux.currentRound();
            if (round < auxRound) {
                round = auxRound;
            }
        }
    }

    /// @return count Token count of the given address up to and including the given round.
    function countTokens(address _address, uint256 _round) public view returns (uint256 count) {
        for (uint256 i = 0; i < auxAddresses.length; i++) {
            Crowdsource aux = Crowdsource(auxAddresses[i]);
            count += aux.countTokens(_address, _round);
        }
    }

    /// @return count Total number of tokens up to and including the given round.
    function countTotalTokens(uint256 _round) external view returns (uint256 count) {
        for (uint256 i = 0; i < auxAddresses.length; i++) {
            Crowdsource aux = Crowdsource(auxAddresses[i]);
            count += aux.countTotalTokens(_round);
        }
    }

    function setGenesis(
        bytes32 _cid,
        uint256 _roundDuration,
        uint256 _numTrainers,
        address[] calldata accounts
    ) external {
        genesis = _cid;
        roundDuration = _roundDuration;
        numTrainers = _numTrainers;
        Crowdsource main = Crowdsource(mainAddress);
        main.setGenesis(genesis, roundDuration, numTrainers, accounts);
    }

    function setConsCurTrainer (address[] calldata _address, uint256 _round, address _contractAddress) external {
        // account = _address;
        // round = _round;
        // contractAddress = _contractAddress;
        Crowdsource cont = Crowdsource(_contractAddress);
        cont.setCurTrainer(_address,round);
    }

    // function setConsEvaluator (address _evaluator, address _contractAddress) external {
    //     evaluator = _evaluator;
    //     Crowdsource cont = Crowdsource(_contractAddress);
    //     cont.setEvaluator(evaluator);
    // }

    // function setAuxCurTrainer (address _address, uint256 _round) external {
    //     Crowdsource aux = Crowdsource(auxAddresses);
    //     aux.setCurTrainer(_address,_round);
    // }


    function addAux(address _evaluator, address[] calldata accounts) external {
        require(genesis != 0, "Genesis not set");
        Crowdsource aux = new Crowdsource();
        aux.setGenesis(genesis, roundDuration, numTrainers-1, accounts);
        aux.setEvaluator(_evaluator);
        auxAddresses.push(address(aux));
    }
}
