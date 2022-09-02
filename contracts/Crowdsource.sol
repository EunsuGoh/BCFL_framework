pragma solidity >=0.4.21 <0.7.0;

/// @title Records contributions made to a Crowdsourcing Federated Learning process
/// @author Harry Cai
contract Crowdsource {
    /// @notice Address of contract creator, who evaluates updates
    address public evaluator;

    /// @notice IPFS CID of genesis model
    bytes32 public genesis;

    /// @dev The timestamp when the genesis model was upload
    uint256 internal genesisTimestamp;

    /// @dev Duration of each training round in seconds
    uint256 internal roundDuration;

    /// @dev Number of updates before training round automatically ends. (If 0, always wait the full roundDuration)
    uint256 internal maxNumUpdates;

    /// @dev Total number of seconds skipped when training rounds finish early
    uint256 internal timeSkipped;

    /// @dev The IPFS CIDs of model updates in each round
    mapping(uint256 => bytes32[]) internal updatesInRound;

    /// @dev The round to which each model update belongs
    mapping(bytes32 => uint256) internal updateRound;

    /// @dev The IPFS CIDs of model updates made by each address
    mapping(address => bytes32[]) internal updatesFromAddress;

    /// @dev Whether or not each model update has been evaluated
    mapping(bytes32 => bool) internal tokensAssigned;

    /// @dev The contributivity score for each model update, if evaluated
    mapping(bytes32 => uint256) internal tokens;

    event Log(string _myString , uint256 round);

    /// @notice Constructor. The address that deploys the contract is set as the evaluator.
    constructor() public {
        evaluator = msg.sender;
    }

    modifier evaluatorOnly() {
        require(msg.sender == evaluator, "Not the registered evaluator");
        _;
    }

    /// @return round The index of the current training round.
    function currentRound() public view returns (uint256 round) {
        uint256 timeElapsed = timeSkipped + now - genesisTimestamp;
        round = 1 + (timeElapsed / roundDuration);
    }

    /// @return remaining The number of seconds remaining in the current training round.
    function secondsRemaining() public view returns (uint256 remaining) {
        uint256 timeElapsed = timeSkipped + now - genesisTimestamp;
        remaining = roundDuration - (timeElapsed % roundDuration);
    }

    /// @return The CID's of updates in the given training round.
    function updates(uint256 _round) external view returns (bytes32[] memory) {
        return updatesInRound[_round];
    }

    /// @return count Token count of the given address up to and including the given round.
    function countTokens(address _address, uint256 _round)
        external
        view
        returns (uint256 count)
    {
        bytes32[] memory updates = updatesFromAddress[_address];
        for (uint256 i = 0; i < updates.length; i++) {
            bytes32 update = updates[i];
            if (updateRound[update] <= _round) {
                count += tokens[updates[i]];
            }
        }
    }

    /// @return count Total number of tokens up to and including the given round.
    function countTotalTokens(uint256 _round) external view returns (uint256 count) {
        for (uint256 i = 1; i <= currentRound(); i++) {
            bytes32[] memory updates = updatesInRound[i];
            for (uint256 j = 0; j < updates.length; j++) {
                bytes32 update = updates[j];
                if (updateRound[update] <= _round){
                    count += tokens[updates[j]];
                }
            }
        }
    }

    /// @return Whether the given address made a contribution in the given round.
    function madeContribution(address _address, uint256 _round)
        public
        view
        returns (bool)
    {
        for (uint256 i = 0; i < updatesFromAddress[_address].length; i++) {
            bytes32 update = updatesFromAddress[_address][i];
            if (updateRound[update] == _round) {
                return true;
            }
        }
        return false;
    }

    /// @notice Sets a new evaluator.
    function setEvaluator(address _newEvaluator) external evaluatorOnly() {
        evaluator = _newEvaluator;
    }

    /// @notice Starts training by setting the genesis model. Can only be called once.
    /// @param _cid The CID of the genesis model
    /// @param _roundDuration Number of seconds per training round
    /// @param _maxNumUpdates Number of updates per round before training round automatically ends. (If 0, always wait the full roundDuration)
    /// @dev Does not reset the training process! Deploy a new contract instead.
    function setGenesis(
        bytes32 _cid,
        uint256 _roundDuration,
        uint256 _maxNumUpdates
    ) external evaluatorOnly() {
        require(genesis == 0, "Genesis has already been set");
        genesis = _cid;
        genesisTimestamp = now;
        roundDuration = _roundDuration;
        maxNumUpdates = _maxNumUpdates;
    }

    /// @notice Records a training contribution in the current round.
    function addModelUpdate(bytes32 _cid, uint256 _round) external {
        emit Log("curRound : ", currentRound());
        emit Log("inserted Round : ", _round);
        require(_round > 0, "Cannot add an update for the genesis round");
        require(
            _round >= currentRound(),
            "Cannot add an update for a past round"
        );
        require(
            _round <= currentRound(),
            "Cannot add an update for a future round"
        );
        require(
            !madeContribution(msg.sender, _round),
            "Already added an update for this round"
        );

        updatesInRound[_round].push(_cid);
        updatesFromAddress[msg.sender].push(_cid);
        updateRound[_cid] = _round;

        if (
            maxNumUpdates > 0 && updatesInRound[_round].length >= maxNumUpdates
        ) {
            // Skip to the end of training round
            timeSkipped += secondsRemaining();
        }
    }

    /// @notice Assigns a token count to an update.
    /// @param _cid The update being rewarded
    /// @param _numTokens The number of tokens to award; should be based on marginal value contribution
    function setTokens(bytes32 _cid, uint256 _numTokens)
        external
        evaluatorOnly()
    {
        require(!tokensAssigned[_cid], "Update has already been rewarded");
        tokens[_cid] = _numTokens;
        tokensAssigned[_cid] = true;
    }
}
